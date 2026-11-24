"""notebook 05 credit scoring function: probability of default (PD) -> points score -> risk band -> policy.

Usage:
    from scorecard import ScoreScale, DEFAULT_BANDS, pd_to_score, assign_band, band_table
    scale = ScoreScale()                         # 600 points at 50:1 good:bad odds, 20 points to double the odds
    score = pd_to_score(pd_values, scale)
    band = assign_band(score)                    # 'A' (safest) ... 'E' (riskiest)
    band_table(y, pd_values, band)               # default rate per band with 95% confidence intervals

Everything here is plain numpy/pandas with no fitted state, so the same functions serve the notebook and the API.
"""
from dataclasses import dataclass, asdict, field
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PD_FLOOR, PD_CAP = 1e-6, 1 - 1e-6


# ---------------------------------------------------------------- score scale
@dataclass(frozen=True)
class ScoreScale:
    """Standard points-to-double-the-odds (PDO) scale.

    score = offset + factor * ln(good:bad odds), with factor = pdo / ln 2 and offset chosen so that
    `base_odds` good:bad maps to `base_score`. Higher score = safer borrower. Every `pdo` points doubles the odds.
    """
    base_score: float = 600.0
    base_odds: float = 50.0          # good:bad, i.e. PD = 1 / 51, about 1.96%
    pdo: float = 20.0
    min_score: float = 300.0
    max_score: float = 850.0

    @property
    def factor(self):
        return self.pdo / np.log(2)

    @property
    def offset(self):
        return self.base_score - self.factor * np.log(self.base_odds)


def pd_to_score(pd_values, scale=ScoreScale(), rounded=False):
    """Map PD to points. Clipped to [min_score, max_score].

    rounded=True returns whole points rounded DOWN, so a displayed score always sits in the same band as the exact
    score (band cut-offs are whole numbers): 639.6 shows as 639, band B, never as 640.
    """
    p = np.clip(np.asarray(pd_values, dtype=float), PD_FLOOR, PD_CAP)
    score = scale.offset + scale.factor * np.log((1 - p) / p)
    score = np.clip(score, scale.min_score, scale.max_score)
    return np.floor(score) if rounded else score


def score_to_pd(score, scale=ScoreScale()):
    """Inverse of pd_to_score (exact inside the clipping range)."""
    log_odds_good = (np.asarray(score, dtype=float) - scale.offset) / scale.factor
    return 1 / (1 + np.exp(log_odds_good))


# ---------------------------------------------------------------- bands and policy
@dataclass(frozen=True)
class Band:
    name: str
    min_score: float                 # inclusive lower bound; the riskiest band uses -inf
    label: str
    policy: str


DEFAULT_BANDS = (
    Band('A', 640, 'Very low risk', 'Approve'),
    Band('B', 620, 'Low risk', 'Approve'),
    Band('C', 590, 'Moderate risk', 'Approve with standard limits'),
    Band('D', 550, 'High risk', 'Refer: manual review, smaller amount or risk-based price'),
    Band('E', -np.inf, 'Very high risk', 'Decline'),
)

# Overlays that override the band policy (checked in this order).
OVERLAYS = {
    'new_market': 'Refer: market without local outcome data (pilot only)',
    'unknown_code': 'Refer: lender or product not seen in training',
}


def assign_band(score, bands=DEFAULT_BANDS):
    """Band name for each score. Bands must be ordered from safest (highest min_score) to riskiest."""
    score = np.asarray(score, dtype=float)
    mins = [b.min_score for b in bands]
    assert all(a > b for a, b in zip(mins, mins[1:])), 'bands must be sorted by descending min_score'
    out = np.full(score.shape, bands[-1].name, dtype=object)
    for b in reversed(bands[:-1]):           # safest band wins
        out[score >= b.min_score] = b.name
    return out


def band_bounds(bands=DEFAULT_BANDS, scale=ScoreScale()):
    """Score and PD range of each band."""
    rows = []
    for i, b in enumerate(bands):
        upper = bands[i - 1].min_score if i else scale.max_score
        lower = max(b.min_score, scale.min_score)
        rows.append({'band': b.name, 'label': b.label, 'score_from': lower, 'score_to': upper,
                     'pd_from': float(score_to_pd(upper, scale)), 'pd_to': float(score_to_pd(lower, scale)),
                     'policy': b.policy})
    return pd.DataFrame(rows).set_index('band')


def policy_for(band, overlay=None, bands=DEFAULT_BANDS):
    """Decision text for one band, with an optional overlay key from OVERLAYS taking precedence."""
    if overlay:
        return OVERLAYS[overlay]
    return {b.name: b.policy for b in bands}[band]


# ---------------------------------------------------------------- validation statistics
def wilson_interval(defaults, n, z=1.96):
    """95% Wilson score interval for a default rate (well behaved for rare events and small bands)."""
    defaults, n = np.asarray(defaults, float), np.asarray(n, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = defaults / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def band_table(y, pd_values, band, exposure=None, bands=DEFAULT_BANDS):
    """Loans, defaults, predicted vs actual default rate and 95% interval per band, safest band first.

    With `exposure` (money at risk per loan), also expected loss (PD x exposure, LGD assumed 100%)
    and realised loss (exposure of loans that defaulted).
    """
    df = pd.DataFrame({'y': np.asarray(y, float), 'pd': np.asarray(pd_values, float), 'band': np.asarray(band)})
    if exposure is not None:
        df['ead'] = np.asarray(exposure, float)
    order = [b.name for b in bands]
    g = df.groupby('band')
    t = pd.DataFrame({'loans': g.size(), 'defaults': g['y'].sum(), 'predicted_pd': g['pd'].mean(),
                      'default_rate': g['y'].mean()}).reindex(order)
    t['loans'] = t['loans'].fillna(0).astype(int)
    t['defaults'] = t['defaults'].fillna(0)
    t['share_of_loans'] = t['loans'] / t['loans'].sum()
    t['share_of_defaults'] = t['defaults'] / max(t['defaults'].sum(), 1)
    t['ci_low'], t['ci_high'] = wilson_interval(t['defaults'], t['loans'].replace(0, np.nan))
    if exposure is not None:
        t['exposure'] = g['ead'].sum().reindex(order)
        t['expected_loss'] = df.assign(el=df['pd'] * df['ead']).groupby('band')['el'].sum().reindex(order)
        t['realised_loss'] = df.assign(rl=df['y'] * df['ead']).groupby('band')['rl'].sum().reindex(order)
    t.index.name = 'band'
    return t


def is_monotonic(table, col='default_rate'):
    """True if the default rate rises from the safest to the riskiest band (empty bands ignored)."""
    v = table[col].dropna().to_numpy()
    return bool(np.all(np.diff(v) > 0))


def adjacent_overlaps(table):
    """Pairs of neighbouring bands whose 95% intervals overlap, i.e. not clearly separated."""
    t = table.dropna(subset=['default_rate'])
    names = t.index.tolist()
    return [(a, b) for a, b in zip(names, names[1:]) if t.loc[b, 'ci_low'] <= t.loc[a, 'ci_high']]


def cumulative_approval(table):
    """If we approve bands from the safest down to each band: approval rate, default rate of the approved
    book, and share of all defaults that would be declined."""
    t = table.fillna({'defaults': 0})
    cum_loans, cum_def = t['loans'].cumsum(), t['defaults'].cumsum()
    return pd.DataFrame({'approve_down_to': t.index, 'approval_rate': cum_loans / t['loans'].sum(),
                         'approved_default_rate': cum_def / cum_loans,
                         'defaults_declined': 1 - cum_def / max(t['defaults'].sum(), 1)}).set_index('approve_down_to')


def gini_ks(y, score):
    """Gini (= 2 x AUC - 1) and Kolmogorov-Smirnov statistic; higher score must mean safer."""
    y = np.asarray(y); s = np.asarray(score, float)
    auc = roc_auc_score(y, -s)
    order = np.argsort(s)
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    return {'auc': float(auc), 'gini': float(2 * auc - 1), 'ks': float(np.max(np.abs(cum_bad - cum_good)))}


def psi(expected_bands, actual_bands, bands=DEFAULT_BANDS, eps=1e-4):
    """Population stability index between two band distributions.
    Rule of thumb: < 0.10 stable, 0.10-0.25 watch, > 0.25 significant shift."""
    order = [b.name for b in bands]
    e = pd.Series(expected_bands).value_counts(normalize=True).reindex(order).fillna(0).clip(lower=eps)
    a = pd.Series(actual_bands).value_counts(normalize=True).reindex(order).fillna(0).clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())


# ---------------------------------------------------------------- persistence (used by the API)
def to_config(scale=ScoreScale(), bands=DEFAULT_BANDS):
    return {'scale': asdict(scale),
            'bands': [{'name': b.name, 'min_score': (None if np.isinf(b.min_score) else b.min_score),
                       'label': b.label, 'policy': b.policy} for b in bands],
            'overlays': OVERLAYS}


def save_config(path, scale=ScoreScale(), bands=DEFAULT_BANDS):
    with open(path, 'w') as f:
        json.dump(to_config(scale, bands), f, indent=2)


def load_config(path):
    cfg = json.load(open(path))
    scale = ScoreScale(**cfg['scale'])
    bands = tuple(Band(b['name'], -np.inf if b['min_score'] is None else b['min_score'], b['label'], b['policy'])
                  for b in cfg['bands'])
    return scale, bands
