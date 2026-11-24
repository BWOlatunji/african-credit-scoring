"""Monthly monitoring for the deployed model (see MONITORING.md for the routine and the action thresholds).

    python -m credit_scoring.monitoring scored_loans.csv [--outcomes]

`scored_loans.csv` is the API's decision log for the period: the model features (as returned with
`include_features=true`), pd, pd_raw, score and band, plus `target` once loans have matured.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring.scorecard import DEFAULT_BANDS, band_table, psi

PSI_WATCH, PSI_ACT = 0.10, 0.25
EPS = 1e-4


def numeric_psi(values, edges, missing_share_ref):
    """PSI of a numeric feature against training deciles; missing values form their own bin."""
    v = pd.Series(values, dtype=float)
    inner = np.unique(edges[1:-1])
    bins = np.digitize(v.dropna(), inner)
    ref = np.full(len(inner) + 1, (1 - missing_share_ref) / (len(inner) + 1))
    act = np.bincount(bins, minlength=len(inner) + 1) / max(len(v), 1)
    ref = np.append(ref, missing_share_ref)
    act = np.append(act, v.isna().mean())
    ref, act = np.clip(ref, EPS, None), np.clip(act, EPS, None)
    return float(((act - ref) * np.log(act / ref)).sum())


def status(value):
    return 'act' if value > PSI_ACT else ('watch' if value > PSI_WATCH else 'stable')


def drift_report(scored: pd.DataFrame, metadata: dict):
    ref = metadata['reference']
    rows = []
    for c, edges in ref['numeric_decile_edges'].items():
        if c in scored:
            v = numeric_psi(scored[c], edges, ref['numeric_missing_share'][c])
            rows.append({'item': c, 'psi': v, 'status': status(v)})
    ref_bands = np.repeat([b for b in ref['band_distribution']],
                          np.round(np.array(list(ref['band_distribution'].values())) * 100000).astype(int))
    v = psi(ref_bands, scored['band'])
    rows.append({'item': 'score band', 'psi': v, 'status': status(v)})
    for c in metadata['categorical_features']:
        if c in scored:
            unseen = (~scored[c].astype(str).isin(metadata['seen_in_training'][c])).mean()
            rows.append({'item': f'{c}: share unseen in training', 'psi': float(unseen),
                         'status': 'act' if unseen > 0.05 else ('watch' if unseen > 0 else 'stable')})
    return pd.DataFrame(rows)


def outcome_report(scored: pd.DataFrame, validation_auc: float):
    """For matured loans: ranking, level and band-by-band calibration."""
    y, p = scored['target'].to_numpy(), scored['pd'].to_numpy()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')
    bt = band_table(y, p, scored['band'])
    summary = {'loans': int(len(y)), 'defaults': int(y.sum()), 'actual_rate': float(y.mean()),
               'mean_pd': float(p.mean()), 'calibration_gap_points': float(abs(p.mean() - y.mean()) * 100),
               'roc_auc': auc, 'auc_drop_vs_validation': float(validation_auc - auc),
               'actions': []}
    if summary['calibration_gap_points'] > 1.0:
        summary['actions'].append('Recalibrate: mean PD differs from the actual default rate by more than 1 point')
    if summary['auc_drop_vs_validation'] > 0.05:
        summary['actions'].append('Review/retrain: ROC AUC on matured loans is more than 0.05 below validation')
    return summary, bt


def main(csv, artifacts='artifacts', outcomes=False):
    metadata = json.loads((Path(artifacts) / 'metadata.json').read_text())
    scored = pd.read_csv(csv)
    drift = drift_report(scored, metadata)
    print('Input and score drift (PSI: <0.10 stable, 0.10-0.25 watch, >0.25 act)')
    print(drift.round(3).to_string(index=False))
    if outcomes:
        summary, bt = outcome_report(scored.dropna(subset=['target']), metadata['validation']['scheme_B_oof_roc_auc'])
        print(json.dumps(summary, indent=2))
        print(bt.round(4).to_string())
    return drift


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv'); ap.add_argument('--artifacts', default='artifacts')
    ap.add_argument('--outcomes', action='store_true', help='CSV includes matured outcomes in column target')
    a = ap.parse_args()
    main(a.csv, a.artifacts, a.outcomes)
