"""Turn the competition files into API-style requests: each loan plus the customer's other loans as history."""
from pathlib import Path

import pandas as pd

from credit_scoring.features import Application, PriorLoan

ROOT = Path(__file__).resolve().parents[1]


def load_pooled():
    train = pd.read_csv(ROOT / 'Train.csv', parse_dates=['disbursement_date'])
    test = pd.read_csv(ROOT / 'Test.csv', parse_dates=['disbursement_date'])
    pooled = pd.concat([train.assign(_set='train'), test.assign(_set='test')], ignore_index=True)
    return train, test, pooled


def history_index(pooled):
    """customer_id -> list of PriorLoan in pooled order (train rows first), one entry per row."""
    out = {}
    for r in pooled[['customer_id', 'tbl_loan_id', 'disbursement_date', 'Total_Amount']].itertuples(index=False):
        out.setdefault(r.customer_id, []).append(
            PriorLoan(str(r.tbl_loan_id), r.disbursement_date.date(), float(r.Total_Amount)))
    return out


def to_application(row):
    return Application(country_id=row.country_id, lender_id=str(row.lender_id), loan_type=row.loan_type,
                       total_amount=float(row.Total_Amount), duration=int(row.duration),
                       new_versus_repeat=row.New_versus_Repeat, lender_portion_funded=float(row.Lender_portion_Funded),
                       disbursement_date=row.disbursement_date.date(), loan_id=str(row.tbl_loan_id),
                       customer_id=str(row.customer_id))


def requests_for(df, hist):
    return [(to_application(r), hist.get(r.customer_id, [])) for r in df.itertuples(index=False)]
