"""Request and response schemas for the scoring API."""
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_HISTORY = 1000
MAX_BATCH = 1000


class PriorLoanIn(BaseModel):
    model_config = ConfigDict(extra='forbid')
    loan_id: str = Field(..., min_length=1, description='Identifier of an earlier loan (tbl_loan_id)')
    disbursement_date: date
    total_amount: float = Field(..., ge=0, description='Principal disbursed, local currency')


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'example': {
        'application_id': 'app-0001', 'customer_id': '266472', 'loan_id': '375320',
        'country_id': 'Kenya', 'lender_id': '267278', 'loan_type': 'Type_1',
        'total_amount': 8000.0, 'duration': 7, 'new_versus_repeat': 'Repeat Loan',
        'lender_portion_funded': 0.3, 'disbursement_date': '2022-08-30',
        'history': [{'loan_id': '361042', 'disbursement_date': '2022-08-02', 'total_amount': 6000.0},
                    {'loan_id': '366781', 'disbursement_date': '2022-08-16', 'total_amount': 7500.0}]}})

    application_id: Optional[str] = Field(None, description='Your reference, echoed back')
    customer_id: Optional[str] = None
    loan_id: Optional[str] = Field(None, description='If given, history entries with the same id are ignored')
    country_id: str = Field(..., min_length=1, description='Kenya has outcome data; other markets are referred')
    lender_id: str = Field(..., min_length=1)
    loan_type: str = Field(..., min_length=1, description='Product code, e.g. Type_1')
    total_amount: float = Field(..., ge=0, description='Principal to disburse, local currency')
    duration: int = Field(..., gt=0, le=3650, description='Term in days')
    new_versus_repeat: Literal['New Loan', 'Repeat Loan']
    lender_portion_funded: float = Field(..., ge=0, le=1, description='Share of principal funded by this lender')
    disbursement_date: date = Field(..., description='Decision / disbursement date')
    history: List[PriorLoanIn] = Field(default_factory=list, max_length=MAX_HISTORY,
                                       description="The customer's other loans; only those before the "
                                                   'disbursement date are used')
    include_features: bool = Field(False, description='Return the model features (for logging and monitoring)')


class Reason(BaseModel):
    feature: str
    reason: str
    log_odds_contribution: float


class HistoryUsed(BaseModel):
    history_used: int
    history_ignored_not_prior: int
    history_ignored_duplicate: int


class ScoreResponse(BaseModel):
    application_id: Optional[str]
    pd: float = Field(..., description='Probability of default after calibration')
    pd_raw: float = Field(..., description='Model output before calibration (log this for recalibration)')
    score: int = Field(..., description='Points: higher is safer; +20 points doubles the odds')
    band: str
    band_label: str
    decision: str
    overlay: Optional[str] = Field(None, description='new_market or unknown_code when an overlay overrides the band')
    reasons: List[Reason] = Field(..., description='Top factors increasing risk for this loan')
    history: HistoryUsed
    model_version: str
    features: Optional[dict] = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    applications: List[ScoreRequest] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: List[ScoreResponse]
