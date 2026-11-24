import numpy as np
import pandas as pd
import pytest

from conftest import needs_artifacts, needs_data
from credit_scoring.recalibrate import apply_calibration, fit_platt


def test_identity_calibration_is_a_no_op():
    p = np.array([0.001, 0.02, 0.5])
    assert np.array_equal(apply_calibration(p, {'method': 'identity'}), p)


def test_platt_recovers_a_shifted_level():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.01, 0.3, 20000)
    y = rng.uniform(size=true_p.size) < true_p
    too_low = true_p / 3                                            # model under-predicts the new period
    cal = fit_platt(too_low, y)
    assert apply_calibration(too_low, cal).mean() == pytest.approx(y.mean(), abs=0.005)
    fixed = apply_calibration(too_low, cal)
    assert np.all(np.diff(fixed[np.argsort(too_low)]) >= 0)       # ranking preserved


def test_platt_refuses_too_few_defaults():
    with pytest.raises(ValueError):
        fit_platt(np.full(100, 0.1), np.r_[np.ones(5), np.zeros(95)])


@needs_data
@needs_artifacts
def test_drift_report_flags_ghana_not_kenya(pipeline):
    from credit_scoring.monitoring import drift_report
    from helpers import history_index, load_pooled, requests_for
    train, test, pooled = load_pooled()
    hist = history_index(pooled)
    rows = {}
    for country in ['Kenya', 'Ghana']:
        part = test[test.country_id == country]
        res = pipeline.score_many(requests_for(part, hist), with_features=True)
        rows[country] = pd.DataFrame([{**r.features, 'band': r.band} for r in res])
    ke = drift_report(rows['Kenya'], pipeline.metadata).set_index('item')
    gh = drift_report(rows['Ghana'], pipeline.metadata).set_index('item')
    assert ke.loc['score band', 'status'] == 'stable'
    assert gh.loc['score band', 'status'] == 'act'
    assert gh.loc['loan_type: share unseen in training', 'status'] == 'act'
