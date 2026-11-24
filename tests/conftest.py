import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ARTIFACTS = ROOT / 'artifacts'
HAS_DATA = (ROOT / 'Train.csv').exists() and (ROOT / 'Test.csv').exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason='competition data (Train.csv, Test.csv) not available')
needs_artifacts = pytest.mark.skipif(not (ARTIFACTS / 'model.txt').exists(),
                                     reason='run python -m credit_scoring.train first')


@pytest.fixture(scope='session')
def pipeline():
    from credit_scoring.pipeline import ScoringPipeline
    return ScoringPipeline.load(ARTIFACTS)
