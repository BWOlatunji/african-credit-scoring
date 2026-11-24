from datetime import date

import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import Application, PriorLoan, build_frame, features_for

APP = Application(country_id='Kenya', lender_id='267278', loan_type='Type_1', total_amount=1000.0, duration=7,
                  new_versus_repeat='Repeat Loan', lender_portion_funded=0.3, disbursement_date=date(2022, 9, 1),
                  loan_id='L9')


def test_no_history():
    f, info = features_for(APP, [])
    assert f['hist_n_prior'] == 0 and np.isnan(f['hist_days_since_last']) and np.isnan(f['hist_log_amount_vs_prior'])
    assert info == {'history_used': 0, 'history_ignored_not_prior': 0, 'history_ignored_duplicate': 0}


def test_only_strictly_earlier_loans_count():
    hist = [PriorLoan('L1', date(2022, 8, 1), 500.0),
            PriorLoan('L2', date(2022, 8, 25), 1000.0),
            PriorLoan('L3', date(2022, 9, 1), 9999.0),     # same day: not prior
            PriorLoan('L4', date(2022, 9, 20), 9999.0)]    # future: never used
    f, info = features_for(APP, hist)
    assert f['hist_n_prior'] == 2
    assert f['hist_days_since_last'] == 7
    assert f['hist_log_amount_vs_prior'] == pytest.approx(np.log1p(1000) - np.mean([np.log1p(500), np.log1p(1000)]))
    assert info['history_ignored_not_prior'] == 2


def test_split_loans_and_own_loan_are_counted_once_or_not_at_all():
    hist = [PriorLoan('L1', date(2022, 8, 1), 500.0), PriorLoan('L1', date(2022, 8, 1), 500.0),
            PriorLoan('L9', date(2022, 8, 2), 700.0)]       # same id as the application
    f, info = features_for(APP, hist)
    assert f['hist_n_prior'] == 1 and info['history_ignored_duplicate'] == 2


def test_future_history_cannot_change_the_features():
    base = [PriorLoan('L1', date(2022, 8, 1), 500.0)]
    f1, _ = features_for(APP, base)
    f2, _ = features_for(APP, base + [PriorLoan('L5', date(2023, 1, 1), 1e6)])
    assert f1 == pytest.approx(f2, nan_ok=True)


def test_new_versus_repeat_is_validated():
    bad = Application(**{**APP.__dict__, 'new_versus_repeat': 'Returning'})
    with pytest.raises(ValueError):
        features_for(bad, [])


def test_unknown_category_becomes_missing():
    f, _ = features_for(Application(**{**APP.__dict__, 'loan_type': 'Type_99'}), [])
    X = build_frame([f], list(f), {'lender_id': ['267278'], 'loan_type': ['Type_1']})
    assert pd.isna(X.loc[0, 'loan_type']) and X.loc[0, 'lender_id'] == '267278'
    assert list(X['loan_type'].cat.categories) == ['Type_1']
