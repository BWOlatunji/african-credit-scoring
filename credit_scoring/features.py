"""Production features for ONE loan application and the customer's earlier loans.

This must reproduce `src/features.py` (the batch pipeline the model was trained on) exactly for the production
feature set; `tests/test_parity.py` checks it on thousands of real rows. The rules carried over from notebook 03:

* history uses only loans disbursed STRICTLY before the application date (same-day loans are not "prior");
* a loan split across several lenders counts once (history is de-duplicated by loan id);
* no label-based history: outcomes of earlier loans are not used (production set P1 is label-free);
* nothing derived from amounts to be repaid is used (those fields likely include post-default charges).
"""
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

NEW_VERSUS_REPEAT = {'New Loan': 1.0, 'Repeat Loan': 0.0}


@dataclass(frozen=True)
class PriorLoan:
    loan_id: str
    disbursement_date: date
    total_amount: float


@dataclass(frozen=True)
class Application:
    country_id: str
    lender_id: str
    loan_type: str
    total_amount: float
    duration: int
    new_versus_repeat: str
    lender_portion_funded: float
    disbursement_date: date
    loan_id: Optional[str] = None
    customer_id: Optional[str] = None


def history_features(app: Application, history: Iterable[PriorLoan]):
    """hist_n_prior, hist_days_since_last, hist_log_amount_vs_prior, plus counts of ignored history entries."""
    seen, prior, ignored_future, ignored_duplicate = set(), [], 0, 0
    for h in history:
        if h.loan_id in seen or (app.loan_id is not None and h.loan_id == app.loan_id):
            ignored_duplicate += 1
            continue
        seen.add(h.loan_id)
        if h.disbursement_date < app.disbursement_date:
            prior.append(h)
        else:
            ignored_future += 1                     # same-day or later loans are not history
    n = len(prior)
    out = {'hist_n_prior': float(n), 'hist_days_since_last': np.nan, 'hist_log_amount_vs_prior': np.nan}
    if n:
        last = max(h.disbursement_date for h in prior)
        out['hist_days_since_last'] = float((app.disbursement_date - last).days)
        out['hist_log_amount_vs_prior'] = float(
            np.log1p(app.total_amount) - np.mean([np.log1p(h.total_amount) for h in prior]))
    return out, {'history_used': n, 'history_ignored_not_prior': ignored_future,
                 'history_ignored_duplicate': ignored_duplicate}


def application_features(app: Application):
    if app.new_versus_repeat not in NEW_VERSUS_REPEAT:
        raise ValueError(f"new_versus_repeat must be one of {sorted(NEW_VERSUS_REPEAT)}")
    return {
        'log_amount_lcu': float(np.log1p(app.total_amount)),
        'duration': float(app.duration),
        'is_new': NEW_VERSUS_REPEAT[app.new_versus_repeat],
        'lender_share': float(app.lender_portion_funded),
        'lender_id': str(app.lender_id),
        'loan_type': str(app.loan_type),
    }


def build_frame(rows: Sequence[dict], feature_order: Sequence[str], categories: dict):
    """Model-ready DataFrame: columns in training order, categoricals with the TRAINING category list
    (a code missing from that list becomes NaN, which LightGBM treats as missing)."""
    X = pd.DataFrame(list(rows))[list(feature_order)]
    for col, cats in categories.items():
        values = X[col].astype(str)
        X[col] = pd.Categorical(values.where(values.isin(cats)), categories=cats)   # unseen code -> missing
    for col in X.columns:
        if col not in categories:
            X[col] = X[col].astype(float)
    return X


def features_for(app: Application, history: Iterable[PriorLoan]):
    hist, info = history_features(app, history)
    return {**application_features(app), **hist}, info
