# Monitoring plan — P1 scoring API

Why this matters here: Notebook 04 showed in-time calibration does not carry forward (2023-24 loans defaulted at 9.5%
while the pre-2023 model predicted 7.1%), and notebook 05 showed bands A-C stop separating on the post-2022 product mix.
The model must be watched and recalibrated, not just deployed.

## What to log for every decision
`application_id`, timestamp, `model_version`, the model features (request with `include_features: true`), `pd_raw`,
`pd`, `score`, `band`, `overlay`, `decision`, and the final human decision. Later join the outcome (`target`) when the
loan matures. Do not log names or contact details; the API logs only band, score, PD and overlay.

## Monthly routine
| Check | How | Stable | Watch | Act |
| --- | --- | --- | --- | --- |
| Input drift per feature | `python -m credit_scoring.monitoring decisions.csv` (PSI vs training deciles) | < 0.10 | 0.10-0.25 | > 0.25: investigate source, product mix |
| Score-band mix | same report, PSI of bands vs training | < 0.10 | 0.10-0.25 | > 0.25: re-validate bands |
| Unseen lenders/products | same report, share unseen | 0% | > 0% | > 5%: product review, retrain plan |
| Level (matured loans) | `--outcomes`: mean PD vs actual default rate | gap < 1 pt | — | gap > 1 pt: recalibrate |
| Ranking (matured loans) | `--outcomes`: ROC AUC vs validation 0.872 | drop < 0.05 | — | drop > 0.05: retrain and full review |
| Band calibration | `--outcomes` band table | rising A→E | a pair overlaps | not rising: re-cut bands |
| Overrides | share of human decisions differing from `decision` per band | — | > 20% in a band | review policy or model |

## Recalibration
When the level check fails, fit Platt scaling on the most recent 6-12 months of **matured** loans (at least 50
defaults): `python -m credit_scoring.recalibrate matured.csv --from 2024-01-01 --to 2024-06-30`. It writes
`artifacts/calibration.json`; restart the service to load it. Recalibration keeps the ranking and changes `pd` and
therefore score and band; `pd_raw` keeps the model output for the next fit.

## Retraining
Retrain (`python -m credit_scoring.train` on the extended data, then the notebook 02 protocol and `pytest`) when ranking
drops by more than 0.05 AUC, a new product exceeds 5% of volume, or quarterly at the latest. A new model gets a new
`model_version`; run old and new side by side on the same applications for one month before switching.

## Ghana and other new markets
All applications are referred (`new_market` overlay). Collect outcomes from the pilot; once there are at least 300
matured Ghana loans with 50+ defaults, measure AUC and calibration on them and decide whether to recalibrate per
country, add a currency-neutral amount, or train with Ghana data.
