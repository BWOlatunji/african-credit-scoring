"""Refit the PD level on recent matured loans (notebook 04: in-time calibration does not carry forward).

    python -m credit_scoring.recalibrate scored_outcomes.csv --from 2024-01-01 --to 2024-06-30

The CSV needs columns pd_raw (the model's raw PD, as returned by the API) and target (1 = defaulted), plus
disbursement_date when a date window is given. Platt scaling keeps the ranking and moves the level; it is fitted
without a penalty (C=inf), and refused on too few defaults. Writes artifacts/calibration.json, which the API loads at start-up.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MIN_DEFAULTS = 50


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(pd_raw, target, min_defaults=MIN_DEFAULTS):
    y = np.asarray(target).astype(int)
    if y.sum() < min_defaults or (len(y) - y.sum()) < min_defaults:
        raise ValueError(f"need at least {min_defaults} defaults and non-defaults to recalibrate; got {int(y.sum())}")
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(_logit(pd_raw).reshape(-1, 1), y)
    return {'method': 'platt', 'slope': float(lr.coef_[0, 0]), 'intercept': float(lr.intercept_[0])}


def apply_calibration(pd_raw, calibration):
    pd_raw = np.asarray(pd_raw, float)
    if calibration is None or calibration.get('method', 'identity') == 'identity':
        return pd_raw
    if calibration['method'] == 'platt':
        z = calibration['slope'] * _logit(pd_raw) + calibration['intercept']
        return 1 / (1 + np.exp(-z))
    raise ValueError(f"unknown calibration method {calibration['method']!r}")


def main(csv, date_from=None, date_to=None, out='artifacts/calibration.json'):
    df = pd.read_csv(csv)
    if date_from or date_to:
        d = pd.to_datetime(df['disbursement_date'])
        df = df[(d >= (date_from or d.min())) & (d <= (date_to or d.max()))]
    cal = fit_platt(df['pd_raw'], df['target'])
    cal['fitted_on'] = {'file': str(csv), 'from': date_from, 'to': date_to, 'loans': int(len(df)),
                        'defaults': int(df['target'].sum()), 'actual_rate': float(df['target'].mean()),
                        'mean_pd_raw': float(df['pd_raw'].mean()),
                        'mean_pd_recalibrated': float(apply_calibration(df['pd_raw'], cal).mean())}
    cal['fitted_at_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    Path(out).write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
    return cal


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv'); ap.add_argument('--from', dest='date_from'); ap.add_argument('--to', dest='date_to')
    ap.add_argument('--out', default='artifacts/calibration.json')
    a = ap.parse_args()
    main(a.csv, a.date_from, a.date_to, a.out)
