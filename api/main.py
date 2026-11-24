"""FastAPI service for the production credit scoring model (P1 + notebook 05 scorecard).

Run locally:   uvicorn api.main:app --reload            (from the project root, after python -m credit_scoring.train)
Docs:          http://127.0.0.1:8000/docs
Artifacts dir: env ARTIFACTS_DIR (default ./artifacts)
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from api.schemas import BatchRequest, BatchResponse, ScoreRequest, ScoreResponse
from credit_scoring import __version__
from credit_scoring.features import Application, PriorLoan
from credit_scoring.pipeline import ScoringPipeline

log = logging.getLogger('credit_api')
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'), format='%(asctime)s %(levelname)s %(name)s %(message)s')
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE['pipeline'] = ScoringPipeline.load(os.getenv('ARTIFACTS_DIR', 'artifacts'))
    log.info('loaded model %s', STATE['pipeline'].metadata['model_version'])
    yield
    STATE.clear()


app = FastAPI(title='African Credit Scoring API', version=__version__, lifespan=lifespan,
              description='Probability of default, points score, risk band and decision for a loan application. '
                          'Production model P1: no fields derived from amounts to be repaid.')


def _pipeline() -> ScoringPipeline:
    pipe = STATE.get('pipeline')
    if pipe is None:
        raise HTTPException(503, 'model not loaded')
    return pipe


def _to_domain(req: ScoreRequest):
    app_ = Application(country_id=req.country_id, lender_id=req.lender_id, loan_type=req.loan_type,
                       total_amount=req.total_amount, duration=req.duration, new_versus_repeat=req.new_versus_repeat,
                       lender_portion_funded=req.lender_portion_funded, disbursement_date=req.disbursement_date,
                       loan_id=req.loan_id, customer_id=req.customer_id)
    history = [PriorLoan(h.loan_id, h.disbursement_date, h.total_amount) for h in req.history]
    return app_, history


def _respond(req: ScoreRequest, result) -> ScoreResponse:
    body = result.to_dict()
    if not req.include_features:
        body['features'] = None
    return ScoreResponse(application_id=req.application_id, **body)


@app.middleware('http')
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers['X-Process-Time-Ms'] = f'{(time.perf_counter() - start) * 1000:.1f}'
    return response


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse(url='/docs')


@app.get('/health')
def health():
    pipe = STATE.get('pipeline')
    return {'status': 'ok' if pipe else 'loading', 'model_version': pipe.metadata['model_version'] if pipe else None}


@app.get('/model-info')
def model_info():
    return _pipeline().info()


@app.post('/score', response_model=ScoreResponse)
def score(req: ScoreRequest):
    pipe = _pipeline()
    app_, history = _to_domain(req)
    result = pipe.score(app_, history, with_features=req.include_features)
    # Decision log without personal data: monitoring needs band, PD and the reason for overrides.
    log.info('scored application_id=%s band=%s score=%s pd=%.5f overlay=%s', req.application_id, result.band,
             result.score, result.pd, result.overlay)
    return _respond(req, result)


@app.post('/score/batch', response_model=BatchResponse)
def score_batch(batch: BatchRequest):
    pipe = _pipeline()
    items = [_to_domain(r) for r in batch.applications]
    with_features = any(r.include_features for r in batch.applications)
    results = pipe.score_many(items, with_features=with_features)
    log.info('scored batch of %d', len(results))
    return BatchResponse(results=[_respond(r, res) for r, res in zip(batch.applications, results)])
