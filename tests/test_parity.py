"""Training-serving parity: the API pipeline must produce exactly the features and PDs of the research code."""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, needs_artifacts, needs_data
from helpers import history_index, load_pooled, requests_for

pytestmark = [needs_data, needs_artifacts]


@pytest.fixture(scope='module')
def data():
    sys.path.insert(0, str(ROOT / 'src'))
    from features import build_features
    train, test, pooled = load_pooled()
    tr = pd.read_csv(ROOT / 'Train.csv', parse_dates=['disbursement_date', 'due_date'])
    te = pd.read_csv(ROOT / 'Test.csv', parse_dates=['disbursement_date', 'due_date'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        feats = build_features(tr, te, pd.read_csv(ROOT / 'economic_indicators.csv'))
    return train, test, pooled, feats, history_index(pooled)


def test_artifacts_match_module4_settings(pipeline):
    settings = json.loads((ROOT / 'outputs' / 'model_settings.json').read_text())['production']
    assert pipeline.features == settings['features']
    for k, v in settings['params'].items():
        assert pipeline.metadata['params'][k] == v


def test_feature_parity_on_sampled_rows(pipeline, data):
    from credit_scoring.features import features_for
    train, test, pooled, feats, hist = data
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(pooled), 4000, replace=False))
    sample = pooled.iloc[idx]
    ours = pd.DataFrame([features_for(a, h)[0] for a, h in requests_for(sample, hist)])
    ref = feats.iloc[idx].reset_index(drop=True)
    for col in pipeline.features:
        if col in pipeline.categories:
            assert (ours[col].astype(str).values == ref[col].astype(str).values).all(), col
        else:
            np.testing.assert_allclose(ours[col].astype(float), ref[col].astype(float), rtol=0, atol=1e-9,
                                       equal_nan=True, err_msg=col)


def test_pd_parity_on_every_test_loan(pipeline, data):
    train, test, pooled, feats, hist = data
    results = pipeline.score_many(requests_for(test, hist))
    ours = np.array([r.pd_raw for r in results])
    ref = pd.read_csv(ROOT / 'outputs' / 'scores_p1_test.csv')
    assert (ref['ID'].values == test['ID'].values).all()
    np.testing.assert_allclose(ours, ref['pd_p1'].values, rtol=0, atol=1e-12)


def test_bands_match_module5(pipeline, data):
    path = ROOT / 'outputs' / 'scores_test_banded.csv'
    if not path.exists():
        pytest.skip('run 05_scorecard.ipynb first')
    train, test, pooled, feats, hist = data
    results = pipeline.score_many(requests_for(test, hist))
    m5 = pd.read_csv(path)
    assert (np.array([r.band for r in results]) == m5['band'].values).all()
    assert (np.array([r.score for r in results]) == m5['score'].values).all()
