"""Fit the production model on all labelled loans and write the deployable artifacts.

    python -m credit_scoring.train                 # from the project root; writes ./artifacts

Reproduces the notebook 04 final P1 model exactly (same features from src/features.py, same settings, seed 0),
so the API's PDs equal outputs/scores_p1_test.csv. Artifacts:
    artifacts/model.txt          LightGBM booster (includes the pandas category mapping)
    artifacts/metadata.json      features, categories, codes seen in training, settings, data hash, reference stats
    artifacts/scorecard.json     score scale, bands and policies (from notebook 05)
    artifacts/calibration.json   recalibration of the PD level (identity until fitted on recent outcomes)
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from credit_scoring import __version__
from credit_scoring.scorecard import DEFAULT_BANDS, assign_band, pd_to_score

NUMERIC_DECILES = np.linspace(0, 1, 11)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(root='.', out='artifacts'):
    root, out = Path(root).resolve(), Path(out)
    sys.path.insert(0, str(root / 'src'))
    from features import build_features          # the training feature pipeline (notebook 03)
    from modeling import BASE_PARAMS              # the notebook 04 defaults the tuned settings override

    train = pd.read_csv(root / 'Train.csv', parse_dates=['disbursement_date', 'due_date'])
    test = pd.read_csv(root / 'Test.csv', parse_dates=['disbursement_date', 'due_date'])
    econ = pd.read_csv(root / 'economic_indicators.csv')
    settings = json.loads((root / 'outputs' / 'model_settings.json').read_text())['production']
    cols, params = settings['features'], {**BASE_PARAMS, **settings['params']}

    feats = build_features(train, test, econ)
    X = feats.iloc[:len(train)][cols].reset_index(drop=True)
    y = train['target'].to_numpy()
    model = lgb.LGBMClassifier(random_state=0, verbose=-1, **params).fit(X, y)

    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / 'model.txt'))
    categorical = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype)]
    numeric = [c for c in cols if c not in categorical]

    oof = pd.read_csv(root / 'outputs' / 'scores_p1_oof.csv')
    oof_band = assign_band(pd_to_score(oof['pd_p1']))
    metadata = {
        'model_name': 'P1 production (LightGBM)',
        'model_version': f"p1-{datetime.now(timezone.utc):%Y%m%d}-{sha256(root / 'Train.csv')[:8]}",
        'package_version': __version__,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'features': cols,
        'categorical_features': categorical,
        'categories': {c: [str(v) for v in X[c].cat.categories] for c in categorical},
        'seen_in_training': {'lender_id': sorted(train['lender_id'].astype(str).unique()),
                             'loan_type': sorted(train['loan_type'].astype(str).unique())},
        'markets_with_outcomes': sorted(train['country_id'].unique()),
        'params': params,
        'training_data': {'file': 'Train.csv', 'sha256': sha256(root / 'Train.csv'), 'rows': int(len(train)),
                          'defaults': int(y.sum()),
                          'disbursed_from': str(train['disbursement_date'].min().date()),
                          'disbursed_to': str(train['disbursement_date'].max().date())},
        'validation': {'scheme_B_oof_roc_auc': float(roc_auc_score(oof['target'], oof['pd_p1'])),
                       'scheme_B_oof_mean_pd': float(oof['pd_p1'].mean()),
                       'actual_default_rate': float(y.mean())},
        'reference': {   # training distributions for drift monitoring (PSI)
            'numeric_decile_edges': {c: [float(v) for v in np.nanquantile(X[c], NUMERIC_DECILES)] for c in numeric},
            'numeric_missing_share': {c: float(X[c].isna().mean()) for c in numeric},
            'band_distribution': {b.name: float(np.mean(oof_band == b.name)) for b in DEFAULT_BANDS},
        },
        'library_versions': {'lightgbm': lgb.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                             'scikit-learn': sklearn.__version__},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    shutil.copyfile(root / 'outputs' / 'scorecard.json', out / 'scorecard.json')
    calibration = out / 'calibration.json'
    if not calibration.exists():
        calibration.write_text(json.dumps({'method': 'identity', 'fitted_on': None,
                                           'note': 'Replace with recalibrate.fit_platt(...) on recent matured loans.'},
                                          indent=2))
    print(f"Wrote {out}/model.txt, metadata.json, scorecard.json, calibration.json "
          f"({metadata['model_version']}; {len(cols)} features; trained on {len(train):,} loans).")
    return metadata


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='.', help='project root with Train.csv, Test.csv, src/ and outputs/')
    ap.add_argument('--out', default='artifacts')
    a = ap.parse_args()
    main(a.root, a.out)
