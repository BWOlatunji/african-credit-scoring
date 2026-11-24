import pytest
from fastapi.testclient import TestClient

from conftest import needs_artifacts

pytestmark = needs_artifacts

BASE = {'application_id': 'app-1', 'customer_id': 'c1', 'loan_id': 'L9', 'country_id': 'Kenya',
        'lender_id': '267278', 'loan_type': 'Type_1', 'total_amount': 8000, 'duration': 7,
        'new_versus_repeat': 'Repeat Loan', 'lender_portion_funded': 0.3, 'disbursement_date': '2022-08-30',
        'history': [{'loan_id': 'L1', 'disbursement_date': '2022-08-02', 'total_amount': 6000},
                    {'loan_id': 'L2', 'disbursement_date': '2022-08-16', 'total_amount': 7500}]}


@pytest.fixture(scope='module')
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get('/health').json()['status'] == 'ok'
    info = client.get('/model-info').json()
    assert info['score_scale']['pdo'] == 20 and [b['band'] for b in info['bands']] == list('ABCDE')
    assert 'markup' not in ' '.join(info['features'])           # production model: no repayment-derived fields


def test_score_response_is_consistent(client):
    r = client.post('/score', json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body['pd'] < 1 and 300 <= body['score'] <= 850
    assert body['band'] in 'ABCDE' and body['overlay'] is None
    assert body['history'] == {'history_used': 2, 'history_ignored_not_prior': 0, 'history_ignored_duplicate': 0}
    assert body['application_id'] == 'app-1' and body['features'] is None
    assert all(x['log_odds_contribution'] > 0 for x in body['reasons'])


def test_include_features(client):
    body = client.post('/score', json={**BASE, 'include_features': True}).json()
    assert body['features']['hist_n_prior'] == 2 and body['features']['hist_days_since_last'] == 14


def test_future_history_is_ignored(client):
    future = {**BASE, 'history': BASE['history'] + [{'loan_id': 'L7', 'disbursement_date': '2023-01-01',
                                                      'total_amount': 1e6}]}
    a, b = client.post('/score', json=BASE).json(), client.post('/score', json=future).json()
    assert a['pd'] == b['pd'] and b['history']['history_ignored_not_prior'] == 1


def test_first_loan_reason(client):
    body = client.post('/score', json={**BASE, 'history': []}).json()
    texts = [x['reason'] for x in body['reasons']]
    assert texts.count('No earlier loans on record') <= 1
    assert 'Days since previous loan' not in texts and 'Number of previous loans' not in texts


def test_new_market_is_referred(client):
    body = client.post('/score', json={**BASE, 'country_id': 'Ghana', 'lender_id': '296542',
                                       'loan_type': 'Type_3'}).json()
    assert body['overlay'] == 'new_market' and body['decision'].startswith('Refer')


def test_unknown_code_is_referred(client):
    body = client.post('/score', json={**BASE, 'loan_type': 'Type_99'}).json()
    assert body['overlay'] == 'unknown_code' and body['decision'].startswith('Refer')


@pytest.mark.parametrize('patch', [{'lender_portion_funded': 1.5}, {'total_amount': -1}, {'duration': 0},
                                   {'new_versus_repeat': 'Returning'}, {'disbursement_date': '30/08/2022'},
                                   {'unexpected_field': 1}, {'loan_type': ''}])
def test_invalid_input_is_rejected(client, patch):
    assert client.post('/score', json={**BASE, **patch}).status_code == 422


def test_batch_matches_single(client):
    reqs = [BASE, {**BASE, 'application_id': 'app-2', 'history': []},
            {**BASE, 'application_id': 'app-3', 'country_id': 'Ghana'}]
    batch = client.post('/score/batch', json={'applications': reqs}).json()['results']
    singles = [client.post('/score', json=r).json() for r in reqs]
    assert [b['pd'] for b in batch] == [s['pd'] for s in singles]
    assert [b['application_id'] for b in batch] == ['app-1', 'app-2', 'app-3']


def test_empty_batch_rejected(client):
    assert client.post('/score/batch', json={'applications': []}).status_code == 422


def test_root_redirects_to_docs(client):
    r = client.get('/', follow_redirects=False)
    assert r.status_code == 307 and r.headers['location'] == '/docs'
