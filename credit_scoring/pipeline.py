"""The scoring pipeline the API serves: application + history -> PD -> score -> band -> decision, with reasons.

    from credit_scoring.pipeline import ScoringPipeline
    pipe = ScoringPipeline.load('artifacts')
    result = pipe.score(application, history)            # one loan
    results = pipe.score_many([(application, history), ...])
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from credit_scoring.features import Application, PriorLoan, build_frame, features_for
from credit_scoring.recalibrate import apply_calibration
from credit_scoring.scorecard import OVERLAYS, assign_band, load_config, pd_to_score

NO_HISTORY_TEXT = 'No earlier loans on record'

# Plain-language reason for each feature pushing risk UP (used for adverse-action style explanations).
REASON_TEXT = {
    'log_amount_lcu': 'Loan amount',
    'duration': 'Loan term (days)',
    'is_new': 'First loan with this lender (new customer)',
    'lender_share': 'Share of the loan funded by this lender',
    'lender_id': 'Lending partner',
    'loan_type': 'Loan product',
    'hist_n_prior': 'Number of previous loans',
    'hist_days_since_last': 'Days since previous loan',
    'hist_log_amount_vs_prior': 'Loan size compared with previous loans',
}


@dataclass
class ScoreResult:
    pd: float
    pd_raw: float
    score: int
    band: str
    band_label: str
    decision: str
    overlay: Optional[str]
    reasons: List[dict]
    history: dict
    model_version: str
    features: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class ScoringPipeline:
    def __init__(self, booster, metadata, scale, bands, calibration):
        self.booster, self.metadata, self.scale, self.bands, self.calibration = \
            booster, metadata, scale, bands, calibration
        self.features = metadata['features']
        self.categories = {c: metadata['categories'][c] for c in metadata['categorical_features']}
        self.seen = {k: set(v) for k, v in metadata['seen_in_training'].items()}
        self.markets = set(metadata['markets_with_outcomes'])
        self.band_info = {b.name: b for b in bands}
        missing = set(self.features) - set(booster.feature_name())
        if missing or list(booster.feature_name()) != list(self.features):
            raise ValueError(f'model features do not match metadata: {booster.feature_name()} vs {self.features}')

    @classmethod
    def load(cls, artifacts_dir='artifacts'):
        d = Path(artifacts_dir)
        booster = lgb.Booster(model_file=str(d / 'model.txt'))
        metadata = json.loads((d / 'metadata.json').read_text())
        scale, bands = load_config(d / 'scorecard.json')
        cal_path = d / 'calibration.json'
        calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {'method': 'identity'}
        return cls(booster, metadata, scale, bands, calibration)

    # ------------------------------------------------------------ core
    def predict_raw(self, X):
        return self.booster.predict(X)

    def _overlay(self, app: Application):
        if app.country_id not in self.markets:
            return 'new_market'
        if str(app.lender_id) not in self.seen['lender_id'] or str(app.loan_type) not in self.seen['loan_type']:
            return 'unknown_code'
        return None

    def _reasons(self, X, top=3):
        """Top features pushing this loan's risk UP (TreeSHAP contributions in log-odds). History features that
        are missing because the customer has no earlier loans are explained as such, once."""
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]      # log-odds contribution per feature
        values = X.to_dict('records')
        out = []
        for row, vals in zip(contrib, values):
            reasons, no_history_added = [], False
            for i in np.argsort(-row):
                if row[i] <= 0 or len(reasons) == top:
                    break
                f = self.features[i]
                no_history = pd.isna(vals[f]) or (f == 'hist_n_prior' and vals[f] == 0)
                if f.startswith('hist_') and no_history:
                    if no_history_added:
                        continue
                    no_history_added = True
                    text = NO_HISTORY_TEXT
                else:
                    text = REASON_TEXT.get(f, f)
                reasons.append({'feature': f, 'reason': text, 'log_odds_contribution': round(float(row[i]), 4)})
            out.append(reasons)
        return out

    def score_many(self, items: Sequence[Tuple[Application, Iterable[PriorLoan]]], with_features=False):
        built = [features_for(app, hist) for app, hist in items]
        X = build_frame([f for f, _ in built], self.features, self.categories)
        p_raw = self.predict_raw(X)
        p = apply_calibration(p_raw, self.calibration)
        scores = pd_to_score(p, self.scale, rounded=True)
        bands = assign_band(scores, self.bands)
        reasons = self._reasons(X)
        results = []
        for (app, _), (feats, info), pr, pc, s, b, r in zip(items, built, p_raw, p, scores, bands, reasons):
            overlay = self._overlay(app)
            band = self.band_info[b]
            results.append(ScoreResult(
                pd=float(pc), pd_raw=float(pr), score=int(s), band=b, band_label=band.label,
                decision=OVERLAYS[overlay] if overlay else band.policy, overlay=overlay, reasons=r, history=info,
                model_version=self.metadata['model_version'],
                features=({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in feats.items()}
                          if with_features else {})))
        return results

    def score(self, app: Application, history: Iterable[PriorLoan] = (), with_features=False):
        return self.score_many([(app, list(history))], with_features=with_features)[0]

    def info(self):
        m = self.metadata
        return {'model_name': m['model_name'], 'model_version': m['model_version'],
                'package_version': m['package_version'], 'trained_at_utc': m['trained_at_utc'],
                'features': self.features, 'training_data': m['training_data'], 'validation': m['validation'],
                'markets_with_outcomes': sorted(self.markets), 'calibration': self.calibration,
                'score_scale': {'base_score': self.scale.base_score, 'base_odds': self.scale.base_odds,
                                'pdo': self.scale.pdo, 'min_score': self.scale.min_score,
                                'max_score': self.scale.max_score},
                'bands': [{'band': b.name, 'min_score': (None if np.isinf(b.min_score) else b.min_score),
                           'label': b.label, 'policy': b.policy} for b in self.bands]}
