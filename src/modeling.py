"""notebook 04 modelling helpers: model factories, calibration checks, final fitting.

Usage:
    from modeling import lgbm_factory, logistic_factory, calibration_table, calibration_summary
    ev.model_factory = lgbm_factory(num_leaves=7)                   # choose the model the Evaluator trains
    ev.run(name, cols, schemes=('B',), protocol='v2-screen-B')     # screen fast
    ev.run(name, cols)                                              # confirm with the full nested protocol
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                   reg_lambda=0.0, colsample_bytree=1.0, subsample=1.0, subsample_freq=0)


def platt_recalibrator(p_fit, y_fit):
    """Fit Platt scaling (a logistic curve on the log-odds of the score) and return a function
    that maps raw scores to recalibrated probabilities. Fit it on RECENT outcomes only."""
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression().fit(lo(np.asarray(p_fit)).reshape(-1, 1), np.asarray(y_fit))
    return lambda p: lr.predict_proba(lo(np.asarray(p)).reshape(-1, 1))[:, 1]


def lgbm_factory(**params):
    """Return a zero-argument function that builds a LightGBM model with these settings.

    The Evaluator calls the factory once per fit, so every fold gets a fresh, identical model.
    """
    settings = {**BASE_PARAMS, **params}

    def make():
        return lgb.LGBMClassifier(random_state=0, verbose=-1, **settings)
    return make


def logistic_factory(categorical_cols, C=1.0):
    """Logistic regression with the preprocessing a linear model needs, fitted inside each fold.

    Numeric columns: fill missing with the median, add a was-missing flag, then standardise.
    Categorical columns: one column per category; categories unseen in training are ignored.
    """
    def make():
        numeric = make_column_selector_excluding(categorical_cols)
        pre = ColumnTransformer([
            ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler())]), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ])
        return Pipeline([('pre', pre),
                         ('model', LogisticRegression(C=C, max_iter=2000))])
    return make


def make_column_selector_excluding(excluded):
    """Select every column except the excluded ones (works on DataFrames at fit time)."""
    return lambda X: [c for c in X.columns if c not in excluded]


# ---------------------------------------------------------------- calibration
def calibration_table(y, p, n_bins=10):
    """Group loans into equal-sized bins by predicted PD; compare predicted vs actual default rate."""
    df = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)})
    df['bin'] = pd.qcut(df['p'].rank(method='first'), n_bins, labels=range(1, n_bins + 1))
    t = df.groupby('bin', observed=True).agg(loans=('y', 'size'), predicted_pd=('p', 'mean'),
                                             actual_rate=('y', 'mean'), defaults=('y', 'sum'))
    t['gap'] = t['predicted_pd'] - t['actual_rate']
    return t


def calibration_summary(y, p):
    """Headline calibration and ranking numbers for one set of predictions."""
    y = np.asarray(y); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    t = calibration_table(y, p)
    return {
        'mean_predicted': float(p.mean()),
        'actual_rate': float(y.mean()),
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p)),
        'expected_calibration_error': float(np.average(t['gap'].abs(), weights=t['loans'])),
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
    }


def top_share(y, p, share=0.10):
    """Share of all defaults found in the riskiest `share` of loans."""
    y = np.asarray(y); p = np.asarray(p)
    top = p >= np.quantile(p, 1 - share)
    return float(y[top].sum() / y.sum())


def fold_auc(y, p, fold_ids):
    """ROC AUC computed separately in each fold, to judge whether a ranking gain beats the noise."""
    y = np.asarray(y); p = np.asarray(p); fold_ids = np.asarray(fold_ids)
    return np.array([roc_auc_score(y[fold_ids == k], p[fold_ids == k]) for k in np.unique(fold_ids)])
