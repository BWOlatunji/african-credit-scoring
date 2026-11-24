import numpy as np
import pandas as pd
import pytest

from credit_scoring.scorecard import (DEFAULT_BANDS, ScoreScale, adjacent_overlaps, assign_band, band_table,
                                      is_monotonic, load_config, pd_to_score, psi, save_config, score_to_pd)


def test_scale_anchor_and_pdo():
    s = ScoreScale()
    assert pd_to_score(1 / 51, s) == pytest.approx(600)            # 50:1 good:bad -> 600
    odds_600 = (1 - score_to_pd(600)) / score_to_pd(600)
    odds_620 = (1 - score_to_pd(620)) / score_to_pd(620)
    assert odds_620 / odds_600 == pytest.approx(2)                  # +20 points doubles the odds


def test_score_is_decreasing_and_invertible():
    p = np.linspace(0.001, 0.9, 200)
    sc = pd_to_score(p)
    assert np.all(np.diff(sc) < 0)
    assert np.allclose(score_to_pd(sc), p)


def test_score_is_clipped():
    assert pd_to_score(1e-12) == 850 and pd_to_score(1 - 1e-12) == 300


@pytest.mark.parametrize('score,band', [(850, 'A'), (640, 'A'), (639.99, 'B'), (620, 'B'), (619.5, 'C'),
                                        (590, 'C'), (589.9, 'D'), (550, 'D'), (549, 'E'), (300, 'E')])
def test_band_boundaries(score, band):
    assert assign_band([score])[0] == band


def test_displayed_score_never_changes_band():
    exact = np.linspace(300, 850, 20001)
    p = score_to_pd(exact)
    shown = pd_to_score(p, rounded=True)
    assert np.array_equal(assign_band(pd_to_score(p)), assign_band(shown))


def test_band_table_monotonic_on_synthetic_data():
    rng = np.random.default_rng(0)
    p = 10 ** rng.uniform(-3.3, -0.5, 200000)                     # PDs spread like a real book (0.05%-32%)
    y = rng.uniform(size=p.size) < p
    t = band_table(y, p, assign_band(pd_to_score(p)))
    assert list(t.index) == [b.name for b in DEFAULT_BANDS]
    assert is_monotonic(t) and adjacent_overlaps(t) == []
    assert t.loans.sum() == p.size


def test_psi_zero_for_identical_and_large_for_shift():
    a = np.array(list('AAAABBBCCD'))
    assert psi(a, a) == pytest.approx(0, abs=1e-12)
    assert psi(a, np.array(list('EEEEEEEEED'))) > 0.25


def test_config_round_trip(tmp_path):
    save_config(tmp_path / 'sc.json')
    scale, bands = load_config(tmp_path / 'sc.json')
    assert scale == ScoreScale() and [b.name for b in bands] == [b.name for b in DEFAULT_BANDS]
    assert np.isinf(bands[-1].min_score)
