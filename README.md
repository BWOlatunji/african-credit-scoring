# African Credit Scoring Challenge: from competition model to production API

Predicting loan default for an asset manager that co-funds digital loans in Kenya and Ghana, using data from the
[Zindi African Credit Scoring Challenge](https://zindi.world/competitions/african-credit-scoring-challenge/).
The project goes from data understanding and validation design to a tuned model, a credit scorecard and a tested,
containerised scoring API.

## Results at a glance

| | Competition model (R1) | Production model (P1) |
| --- | --- | --- |
| Purpose | Zindi leaderboard (F1 on a hard 0/1) | Lending decisions: PD, score, risk band |
| Features | 10, including markup from `Total_Amount_to_Repay` | 9, nothing derived from amounts to be repaid |
| Validation (loan-grouped 5-fold, nested threshold) | F1 0.833 ± 0.010 | ROC AUC 0.872 (Gini 0.74); top 10% of scores holds 61% of defaults |
| Forward in time (train < 2023, score 2023-24) | F1 0.578; AUC 0.906 | AUC 0.596; level fixed by recalibration |
| Zindi (late submission) | Public 0.614, private 0.642 | — |
| Scorecard | — | Bands A-E: 0.18% → 19.0% default, calibrated in every band |
| Serving | — | FastAPI + Docker; API PDs match the research code on all 18,594 test loans |

**The central finding:** markup looked like the strongest feature, but within the same product and term, repaid loans
carry the contract price and defaulted loans carry far more (0.7% vs 41.2% for 7-day Type_1 loans). That is the pattern
of charges added after default. It is legitimate for the leaderboard, because the test file is filled the same way,
and unusable at decision time. So the project keeps two tracks: a competition model with markup, and a production
model without it.

## Repository layout

```
01_data_understanding.ipynb  grain (customer → loan → lender share), segments, vintages, the Ghana gap, data quality
02_validation.ipynb          adversarial validation, five validation schemes, fixed folds, nested thresholds
03_features.ipynb            feature families on two tracks; the markup leakage test; sensitivity tests (protocol v1)
03b_confirm_v2.ipynb         key models re-confirmed with the corrected nested evaluator (protocol v2)
04_modelling.ipynb           tuning with guardrails, logistic benchmark, calibration in time and forward, submission
05_scorecard.ipynb           score scale (600 at 50:1, PDO 20), bands A-E, policy, stress tests, test-set application
src/                         research code: features.py, evaluate.py (v2), modeling.py; v1 versions kept for 03
credit_scoring/              production package: features, pipeline, scorecard, train, recalibrate, monitoring
api/                         FastAPI service
artifacts/                   trained production model and its configuration
tests/                       48 tests: scorecard, features, training-serving parity, API, monitoring
outputs/                     folds, experiment log, feature sets, scores, submissions, scorecard tables
docs/                        project review slides
MODEL_CARD.md                intended use, performance, limitations
MONITORING.md                monthly monitoring routine and action thresholds
RUNNING.md                   step-by-step guide: local .venv or Docker on Windows
```

Experiment names in `outputs/experiments.csv` carry the notebook they come from: `M2` = notebook 02, `M3` = 03/03b,
`M4` = 04. The `protocol` column says how each number was produced: `v1` (original evaluator), `v2-nested` (reported
numbers), `v2-screen-B` (fast screening, never reported).

## Data

The competition data is not redistributed here. Download `Train.csv`, `Test.csv` and `SampleSubmission.csv` from the
[competition page](https://zindi.world/competitions/african-credit-scoring-challenge/) and place them in the repository
root. `economic_indicators.csv` (World Bank indicators provided with the competition) is included.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# research notebooks, in order (notebook 04 takes 35-45 minutes)
for nb in 01_data_understanding 02_validation 03_features 03b_confirm_v2 04_modelling 05_scorecard; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done

python -m credit_scoring.train         # writes artifacts/ (about 15 s)
pytest                                 # 48 tests
uvicorn api.main:app --reload          # http://127.0.0.1:8000 opens the API docs
```

Or run the API in Docker: `docker build -t credit-scoring-api .` then `docker run -p 8000:8000 credit-scoring-api`.
Full instructions, request examples and troubleshooting: [RUNNING.md](RUNNING.md).

## Method in brief

1. **Validation first.** Adversarial validation shows Kenya test is the same population as train (AUC 0.50) and Ghana
   is not (AUC 1.00). The primary scheme groups by loan, mirroring how Kenya test was split; guardrails cover new
   customers, forward-in-time loans and held-out lenders (a Ghana proxy). Thresholds are chosen on other folds only, and
   a change counts only if it beats one fold standard deviation or wins 4 of 5 folds without hurting a guardrail.
2. **Features by family.** Each family is scored on the same fixed folds and logged. Only pricing passed on the
   competition track. On the production track, label-free customer history was kept: label-based history makes
   every Ghana borrower look like a first-timer.
3. **Tuning with guardrails.** The best-scoring R1 setting on the primary scheme lost 0.157 F1 on future loans and was
   rejected. P1 uses shallow trees with large leaves, which rank better in all five folds.
4. **Calibration.** Both models are calibrated in time. Forward, the population changed: the safe 7-day Type_1 product
   was only disbursed from July to December 2022. Platt scaling on recent outcomes restores P1's level.
5. **Scorecard.** Five bands on round score cut-offs fixed before looking at outcomes, a policy per band, and overlays
   that refer new markets and unseen lenders or products to manual review.
6. **Production.** A stateless API: each request carries the application and the customer's earlier loans. Parity tests
   prove the API reproduces the research pipeline exactly.

## Known limitations

Forward ranking on the post-2022 product mix is weak (AUC 0.596); band C under-predicts outside the main product;
Ghana has no labels and amounts are in local currency, so every Ghana loan is referred; loss given default is assumed
100%; the client's days-past-due default definition is unknown. Details and next steps are in
[MODEL_CARD.md](MODEL_CARD.md).

## Acknowledgement

Developed during the 3MTT Mentorship Program.
