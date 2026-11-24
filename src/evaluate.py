"""Validation protocol (notebooks 02-03) with nested threshold calibration.

Usage:
    from evaluate import Evaluator, log_experiment
    ev = Evaluator(train, test, feats, folds)
    result = ev.run("R1 baseline + pricing", cols)                       # full protocol (reportable)
    result = ev.run("tuning candidate", cols, schemes=('B',),
                    protocol='v2-screen-B')                              # fast screening (never reported)

Protocol versions written to the experiment log:
    v1           original evaluator (src/evaluate_v1.py), threshold chosen on other folds' OOF
    v2-nested    this file, all schemes B-E: the only numbers to report
    v2-screen-B  this file, scheme B only: for ranking candidates during tuning

Scheme B/C/D F1 values use nested cross-fitting for threshold selection:
the outer validation fold is excluded from both threshold calibration and
model fitting. Scheme E calibrates a separate threshold using only the
non-held-out lenders. Final test threshold is selected from all training OOF
predictions, which is a training-derived operating point, not an unbiased
performance estimate.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import date
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
TIME_CUT = pd.Timestamp('2023-01-01')


def best_threshold(y_true, p):
    """Return the threshold with maximum F1, ignoring non-finite predictions."""
    y = np.asarray(y_true)
    prob = np.asarray(p, dtype=float)
    valid = np.isfinite(prob) & pd.notna(y)
    y, prob = y[valid], prob[valid]
    if len(y) == 0:
        raise ValueError("Cannot select a threshold without valid calibration predictions.")
    scores = [f1_score(y, prob >= t, zero_division=0) for t in THRESHOLDS]
    k = int(np.argmax(scores))
    return float(THRESHOLDS[k]), float(scores[k])


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1
    )


class Evaluator:
    def __init__(self, train, test, feats, folds, model_factory=make_model):
        n = len(train)
        if len(feats) != len(train) + len(test):
            raise ValueError(
                "feats must contain train rows followed by test rows, matching build_features()."
            )
        if len(folds) != n:
            raise ValueError("folds.csv must have exactly one row per training row.")
        if not np.array_equal(folds['ID'].values, train['ID'].values):
            raise ValueError('folds.csv is not in Train.csv row order')

        self.X = feats.iloc[:n].reset_index(drop=True)
        self.X_test = feats.iloc[n:].reset_index(drop=True)
        self.y = train['target'].to_numpy()
        self.fold_loan = folds['fold_loan'].to_numpy()
        self.fold_customer = folds['fold_customer'].to_numpy()
        self.past = (pd.to_datetime(train['disbursement_date']) < TIME_CUT).to_numpy()
        self.lender = train['lender_id'].to_numpy()
        self.test_country = test['country_id'].to_numpy()
        self.model_factory = model_factory

    @staticmethod
    def _positive_probability(model, X):
        """Handle estimators trained on a single class without assuming class order."""
        if hasattr(model, 'classes_') and len(model.classes_) == 1:
            return np.full(len(X), float(model.classes_[0]), dtype=float)
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros(len(X), dtype=float)
        return probs[:, classes.index(1)]

    def _fit_predict(self, X, y, fit_rows, predict_rows):
        if not np.any(fit_rows):
            raise ValueError("A validation split produced no eligible training rows.")
        y_fit = y[fit_rows]
        if len(np.unique(y_fit)) == 1:
            return np.full(int(np.sum(predict_rows)), float(y_fit[0]), dtype=float)
        model = self.model_factory().fit(X.loc[fit_rows] if hasattr(X, 'loc') else X[fit_rows], y_fit)
        X_pred = X.loc[predict_rows] if hasattr(X, 'loc') else X[predict_rows]
        return self._positive_probability(model, X_pred)

    def _crossfit_predictions(self, X, y, fold_ids, eligible_mask):
        """OOF predictions only for eligible rows; no ineligible row is used to fit."""
        eligible = np.asarray(eligible_mask, dtype=bool)
        fold_ids = np.asarray(fold_ids)
        pred = np.full(len(y), np.nan, dtype=float)
        for k in np.unique(fold_ids[eligible]):
            valid = eligible & (fold_ids == k)
            fit = eligible & (fold_ids != k)
            if not np.any(valid):
                continue
            pred[valid] = self._fit_predict(X, y, fit, valid)
        return pred

    def _cv(self, X, y, fold_ids, train_mask):
        """Nested OOF: each outer fold is excluded from threshold calibration and fitting."""
        y = np.asarray(y)
        fold_ids = np.asarray(fold_ids)
        train_mask = np.asarray(train_mask, dtype=bool)
        oof = np.full(len(y), np.nan, dtype=float)
        f1s = []

        for k in np.unique(fold_ids):
            outer_valid = fold_ids == k
            outer_train = ~outer_valid

            # Inner OOF predictions are generated using only the outer-training
            # partition. Rows excluded by train_mask may still be scored/calibrated.
            inner_pred = np.full(len(y), np.nan, dtype=float)
            for j in np.unique(fold_ids[outer_train]):
                inner_valid = outer_train & (fold_ids == j)
                inner_fit = outer_train & (fold_ids != j) & train_mask
                if np.any(inner_valid) and np.any(inner_fit):
                    inner_pred[inner_valid] = self._fit_predict(
                        X, y, inner_fit, inner_valid
                    )

            calibration_rows = outer_train & np.isfinite(inner_pred)
            if not np.any(calibration_rows):
                raise ValueError(
                    f"Cannot calibrate threshold for outer fold {k}: "
                    "no inner out-of-fold predictions were produced."
                )
            threshold, _ = best_threshold(y[calibration_rows], inner_pred[calibration_rows])

            outer_fit = outer_train & train_mask
            if not np.any(outer_fit):
                raise ValueError(f"Outer fold {k} has no eligible training rows.")
            oof[outer_valid] = self._fit_predict(X, y, outer_fit, outer_valid)
            f1s.append(
                f1_score(y[outer_valid], oof[outer_valid] >= threshold, zero_division=0)
            )

        return oof, f1s

    def run(
        self,
        name,
        cols,
        train_mask=None,
        verbose=True,
        schemes=("B", "C", "D", "E"),
        protocol="v2-nested",
    ):
        unknown = set(schemes) - {'B', 'C', 'D', 'E'}
        if unknown:
            raise ValueError(f"Unknown schemes: {sorted(unknown)}")
        X, y = self.X[cols], self.y
        m = np.ones(len(y), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
        if len(m) != len(y):
            raise ValueError("train_mask must have one boolean value per training row.")
        if not np.any(m):
            raise ValueError("train_mask excludes every training row.")

        r = {
            "experiment": name,
            "n_features": len(cols),
            "protocol": protocol,
            "schemes": "".join(schemes),
        }

        # Scheme B: loan-group outer folds with nested threshold calibration.
        oof_B, f1s_B = self._cv(X, y, self.fold_loan, m)
        thr_B, _ = best_threshold(y, oof_B)
        r.update(
            f1_B=float(np.mean(f1s_B)),
            f1_B_sd=float(np.std(f1s_B)),
            roc_auc_B=(float(roc_auc_score(y, oof_B)) if len(np.unique(y)) == 2 else np.nan),
            pr_auc_B=(float(average_precision_score(y, oof_B)) if np.sum(y == 1) else np.nan),
            threshold_B=thr_B
        )
        r['f1_B_folds'] = ' '.join(f'{v:.3f}' for v in f1s_B)

        # Scheme C: customer-group outer folds, same nested threshold protocol.
        if 'C' in schemes:
            _, f1s_C = self._cv(X, y, self.fold_customer, m)
            r['f1_C'] = float(np.mean(f1s_C))
        else:
            r['f1_C'] = np.nan

        # Scheme D: forward-time holdout; threshold calibrated only within the past.
        past, fut = self.past, ~self.past
        if 'D' not in schemes or not np.any(past) or not np.any(fut):
            r.update(f1_D=np.nan, mean_pred_D=np.nan, actual_D=np.nan)
        else:
            oof_past, _ = self._cv(
                X.loc[past].reset_index(drop=True), y[past],
                self.fold_loan[past], m[past]
            )
            thr_past, _ = best_threshold(y[past], oof_past)
            p_fut = self._fit_predict(X, y, past & m, fut)
            r.update(
                f1_D=float(f1_score(y[fut], p_fut >= thr_past, zero_division=0)),
                mean_pred_D=float(np.mean(p_fut)),
                actual_D=float(np.mean(y[fut]))
            )

        # Scheme E: hold out each lender. Calibrate threshold only on other
        # lenders' cross-fitted predictions; never use the held-out lender labels.
        aucs, f1s_E, weights = [], [], []
        for lender_value in (np.unique(self.lender) if 'E' in schemes else []):
            fit_lenders = (self.lender != lender_value) & m
            held_lender = self.lender == lender_value
            if not np.any(held_lender) or not np.any(fit_lenders):
                continue

            calibration_pred = self._crossfit_predictions(
                X, y, self.fold_loan, fit_lenders
            )
            cal_rows = fit_lenders & np.isfinite(calibration_pred)
            if not np.any(cal_rows):
                threshold = thr_B
            else:
                threshold, _ = best_threshold(y[cal_rows], calibration_pred[cal_rows])

            p_held = self._fit_predict(X, y, fit_lenders, held_lender)
            y_held = y[held_lender]
            if len(np.unique(y_held)) == 2:
                aucs.append(float(roc_auc_score(y_held, p_held)))
            else:
                aucs.append(np.nan)
            f1s_E.append(float(f1_score(y_held, p_held >= threshold, zero_division=0)))
            weights.append(int(np.sum(held_lender)))

        valid_aucs = [v for v in aucs if np.isfinite(v)]
        r.update(
            auc_E_mean=(float(np.mean(valid_aucs)) if valid_aucs else np.nan),
            auc_E_min=(float(np.min(valid_aucs)) if valid_aucs else np.nan),
            f1_E_weighted=(float(np.average(f1s_E, weights=weights)) if weights else np.nan)
        )

        # Final test scores use all eligible training rows and the training-OOF threshold.
        if len(self.X_test):
            if len(np.unique(y[m])) == 1:
                p_test = np.full(len(self.X_test), float(y[m][0]), dtype=float)
            else:
                final_model = self.model_factory().fit(X.loc[m], y[m])
                p_test = self._positive_probability(final_model, self.X_test[cols])
        else:
            p_test = np.array([], dtype=float)
        for country in ['Kenya', 'Ghana']:
            country_mask = self.test_country == country
            r[f'flag_rate_{country}'] = (
                float(np.mean(p_test[country_mask] >= thr_B))
                if np.any(country_mask) else np.nan
            )

        self.last_oof_B, self.last_test_prob = oof_B, p_test
        if verbose:
            print(
                f"{name}: F1 B {r['f1_B']:.3f} ± {r['f1_B_sd']:.3f} | "
                f"C {r['f1_C']:.3f} | D {r['f1_D']:.3f} | "
                f"E AUC {r['auc_E_mean']:.3f} | Ghana flag {r['flag_rate_Ghana']:.1%}"
            )
        return r


def log_experiment(row, path=Path('outputs') / 'experiments.csv'):
    """One row per experiment name; re-running replaces the old row."""
    row = {'date': str(date.today()), **row}
    new = pd.DataFrame([row])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if 'experiment' not in old.columns:
            raise ValueError(f"{path} exists but has no 'experiment' column.")
        new = pd.concat(
            [old[old['experiment'] != row['experiment']], new],
            ignore_index=True
        )
    new.to_csv(path, index=False)
    return new
