# Model card — P1 production credit scoring model

**Version:** see `GET /model-info` (`model_version`, e.g. `p1-20261124-20d43743`) · **Package:** `credit_scoring` 1.0.0
**Status:** ready for a supervised pilot in Kenya; **not** for automated decisions.

## Intended use
Estimate the probability of default (PD) of a loan application at the time of decision, convert it into a points score
and risk band (A-E), and recommend a decision per band. Users: the asset manager's credit team, as decision support
with human review of bands D-E and of all overlays.

**Out of scope:** automated decline without review; markets without local outcome data (Ghana is referred to manual
review); pricing or provisioning without recalibration on recent outcomes; any use of the competition model (R1).

## Data
68,654 labelled loan shares (Kenya only), disbursed Oct 2021-Nov 2024, 1.83% default. One row is one lender's share
of one loan. The 7-day Type_1 product is 90% of rows and was only disbursed 12 Jul-1 Dec 2022; from 2023 the book is
other products (mainly Type_7, new in 2023) defaulting at 9.5%. "Default" follows the client's label; the days-past-due
rule was never specified (open question).

## Features (9)
Loan amount (log, local currency), term, new vs repeat, lender's share of the loan, lender and product codes, and
three label-free history features from the customer's earlier loans (count, days since last, size vs earlier loans).
**Excluded on purpose:** everything derived from `Total_Amount_to_Repay` / `Lender_portion_to_be_repaid`: same
product and term show 0.7% markup when repaid vs 41% when defaulted, so these fields very likely include charges
added after default (notebook 03). Label-based history is excluded too (Ghana trap, notebook 03).

## Model
LightGBM, 600 trees, learning rate 0.025, 7 leaves, at least 200 loans per leaf (chosen for ranking; notebook 04).

## Performance (validation protocol: loan-grouped 5-fold, nested thresholds)
| Measure | Value |
| --- | --- |
| ROC AUC / Gini / KS, in time | 0.872 / 0.743 / 0.575 (better than default settings in 5 of 5 folds) |
| Gini within products | Type_1 7-day 0.60; other products 0.46 |
| Riskiest 10% of scores | 61% of all defaults |
| Calibration in time | mean PD 1.82% vs 1.83% actual; every decile within 0.3 points |
| Bands A-E (in time) | 0.18%, 0.85%, 1.7%, 5.3%, 19.0% default; no overlapping 95% intervals |
| Held-out lender AUC | 0.714 (mean of 4) |
| Forward in time (train < 2023, score 2023-24) | AUC 0.596; mean PD 7.1% vs 9.5% actual |
| After recalibration on 2023, tested on 2024 | mean PD 8.4% vs 8.7% actual; calibration gap 5.0 → 2.0 points |

## Known limitations
1. **Ranking degrades on new products.** Forward AUC 0.596; within the post-2022 product mix, bands A-C are not
   distinguishable (6-7% default) and only D-E separate. Recalibration fixes the level, not the ranking.
2. **Band C under-predicts outside the 7-day Type_1 product** (2.0% predicted vs 4.2% actual).
3. **Ghana is unvalidated.** No labels; PSI of the band mix 3.7. Leaderboard probes suggest Ghana defaults at about
   12.6-13.1% vs P1's mean PD of 12.4%: plausible level, unknown ranking. All Ghana applications are referred.
4. **Amounts are in local currency.** The model compares GHS with KES amounts; acceptable only because Ghana is
   referred. A currency-neutral amount is required before any Ghana automation.
5. **Unseen lenders/products** are routed to review; the model treats their codes as missing.
6. **Losses:** expected loss in notebook 05 assumes 100% loss given default (no recovery data).

## Fairness and explainability
No personal or protected attributes are used. Each response lists the top factors increasing risk (TreeSHAP
contributions) for adverse-action explanations. Lender and product codes act on the product, not on the person, but
should be reviewed with the client for indirect effects.

## Monitoring and change control
Monthly routine and action thresholds in `MONITORING.md`. Any change of features, settings or band cut-offs needs a
rerun of the notebook 02 protocol, the parity tests (`pytest`), and a new model version.
