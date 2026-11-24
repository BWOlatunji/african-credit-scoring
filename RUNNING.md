# Running the credit scoring pipeline on Windows (.venv or Docker)

The production pipeline turns a loan application and the customer's earlier loans into a probability of default (PD),
a 300-850 score, a risk band A-E and a decision, served by a FastAPI service. There are two ways to run it, and both
serve the same model from the same `artifacts/` folder. (The commands also work on macOS and Linux; use
`source .venv/bin/activate` to activate.)

| | Option A: local `.venv` | Option B: Docker container |
| --- | --- | --- |
| Best for | Development, notebooks, tests, retraining | Deployment, demos, running on another machine |
| Needs | Python 3.11-3.13 | Docker Desktop with WSL 2 |
| Python packages | Installed in `.venv` | Installed inside the image |
| Can retrain the model | Yes (`python -m credit_scoring.train`) | No: retrain locally, then rebuild the image |
| Auto-reload on code change | Yes (`--reload`) | No: rebuild the image |
| API address | http://127.0.0.1:8000 | http://127.0.0.1:8000 (or the port you map) |

The usual workflow: retrain and test in Option A, then build and ship the image with Option B. Training always happens
in the `.venv`; the API loads `artifacts/` either straight from the folder (A) or from inside the image (B).

## Prerequisites

| What | Needed for | Check it with | Notes |
| --- | --- | --- | --- |
| Python 3.11, 3.12 or 3.13 (64-bit) | Option A | `python --version` | pandas 3.1 needs 3.11 or newer. Tick "Add python.exe to PATH" when installing from python.org |
| Git for Windows | Both | `git --version` | Includes Git Bash; PowerShell works too |
| Microsoft Visual C++ Redistributable (x64) | Option A | Present on most PCs | LightGBM needs it on Windows; install it if `import lightgbm` fails |
| Docker Desktop with the WSL 2 backend | Option B | `docker version` shows a Client and a Server | Start Docker Desktop before any `docker` command |
| `Train.csv`, `Test.csv`, `SampleSubmission.csv` | Training, notebooks, parity tests | Files in the project root | From the Zindi competition page; kept out of git |
| About 2 GB free disk | Both | | Packages about 600 MB; the Docker image about 1 GB |

All commands run from the project root. If the folder path contains spaces, put it in quotes:
`cd "C:\path with spaces\african-credit-scoring"`.

## Option A: run in a local .venv

**1. Open a terminal in the project root.** PowerShell or Git Bash; where they differ, both are shown.

**2. Create the virtual environment (once).**

```powershell
python -m venv .venv
```

**3. Activate it (every new terminal) and install the packages (once).**

```powershell
# PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Git Bash
source .venv/Scripts/activate
```

The prompt now starts with `(.venv)`. Then install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` adds Jupyter for the notebooks; `requirements-dev.txt` adds the API packages, pytest and httpx. For
the API alone, `pip install -r requirements-prod.txt` is enough.

**4. Reproduce the research outputs (optional).** Only needed if `outputs/` is missing or you changed the research
code. Notebook 04 takes 35-45 minutes, notebook 05 about 2.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 04_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_scorecard.ipynb
```

**5. Train the production model.** Writes `artifacts/` in about 15 seconds and reproduces the notebook 04 model
exactly (same features, settings and seed).

```bash
python -m credit_scoring.train
```

**6. Run the tests.** Expect `48 passed` in about 25 seconds. Without `Train.csv` and `Test.csv` the parity tests are
skipped, not failed.

```bash
pytest
```

**7. Start the API.**

```bash
uvicorn api.main:app --reload
```

Wait for `loaded model p1-...` and `Application startup complete`, then open http://127.0.0.1:8000. It redirects to the
interactive docs, where **POST /score** → **Try it out** → **Execute** sends a sample request.

**8. Send a request from a second terminal.**

```powershell
# PowerShell
$body = @{
  application_id = 'app-0001'; country_id = 'Kenya'; lender_id = '267278'; loan_type = 'Type_1'
  total_amount = 8000; duration = 7; new_versus_repeat = 'Repeat Loan'; lender_portion_funded = 0.3
  disbursement_date = '2022-08-30'
  history = @(@{loan_id = '361042'; disbursement_date = '2022-08-02'; total_amount = 6000},
              @{loan_id = '366781'; disbursement_date = '2022-08-16'; total_amount = 7500})
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/score -ContentType 'application/json' -Body $body
```

```bash
# Git Bash / macOS / Linux
curl -s -X POST http://127.0.0.1:8000/score -H "content-type: application/json" -d '{
  "application_id": "app-0001", "country_id": "Kenya", "lender_id": "267278", "loan_type": "Type_1",
  "total_amount": 8000, "duration": 7, "new_versus_repeat": "Repeat Loan", "lender_portion_funded": 0.3,
  "disbursement_date": "2022-08-30",
  "history": [{"loan_id": "361042", "disbursement_date": "2022-08-02", "total_amount": 6000},
              {"loan_id": "366781", "disbursement_date": "2022-08-16", "total_amount": 7500}]}'
```

Expected: `score` 592, `band` C, `decision` "Approve with standard limits". **To stop**, press Ctrl+C in the server
terminal; `deactivate` leaves the `.venv`.

## Option B: run in a Docker container

The image holds Python 3.12, the pinned production packages, the `credit_scoring` and `api` code and the trained
`artifacts/`; nothing else goes in (see `.dockerignore`). It runs as a non-root user with a built-in health check.

1. **Start Docker Desktop** and wait until it says it is running.
2. **Make sure `artifacts/` exists.** It ships with the repository; after retraining, the image picks up the new model
   at build time.
3. **Build the image** (the first build downloads Python and the packages, a few minutes):
   ```bash
   docker build -t credit-scoring-api .
   ```
4. **Run the container** (stop any local server on port 8000 first):
   ```bash
   docker run --rm -d -p 8000:8000 --name credit-api credit-scoring-api
   ```
5. **Check it is healthy:** `docker ps` shows `(healthy)` after about 30 seconds; `docker logs credit-api` shows
   `loaded model p1-...`. Then open http://127.0.0.1:8000 or send the requests from Option A step 8.
6. **Stop it:** `docker stop credit-api`.

| Goal | Command |
| --- | --- |
| Use another port | `docker run --rm -d -p 8080:8000 --name credit-api credit-scoring-api`, then http://127.0.0.1:8080 |
| More detailed logs | add `-e LOG_LEVEL=DEBUG` |
| More worker processes (default 2) | `docker run --rm -d -p 8000:8000 credit-scoring-api uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4` |
| New artifacts without rebuilding, PowerShell | `docker run --rm -d -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts:ro" credit-scoring-api` |
| Same, Git Bash | `MSYS_NO_PATHCONV=1 docker run --rm -d -p 8000:8000 -v "$(pwd -W)/artifacts:/app/artifacts:ro" credit-scoring-api` |
| Shell inside the running container | `docker exec -it credit-api sh` |

The container reads `ARTIFACTS_DIR` (default `/app/artifacts`) and `LOG_LEVEL` (default `INFO`).

## Using the API

| Method and path | What it does |
| --- | --- |
| `GET /` | Redirects to `/docs` |
| `GET /health` | `{"status": "ok", "model_version": "p1-..."}`; used by the Docker health check |
| `GET /model-info` | Model version, features, training data, validation results, score scale, bands and policies |
| `POST /score` | Scores one application |
| `POST /score/batch` | Scores up to 1,000 applications: `{"applications": [ ... ]}` |

**Request fields (`POST /score`)**

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `country_id` | text | yes | `Kenya` is scored normally; any other market is referred (no local outcome data) |
| `lender_id` | text | yes | e.g. `267278`; a lender not seen in training is referred |
| `loan_type` | text | yes | e.g. `Type_1`; a product not seen in training is referred |
| `total_amount` | number | yes | Principal, local currency, 0 or more |
| `duration` | whole number | yes | Term in days, 1-3,650 |
| `new_versus_repeat` | text | yes | `New Loan` or `Repeat Loan` |
| `lender_portion_funded` | number | yes | Share funded by this lender, 0-1 |
| `disbursement_date` | date | yes | `YYYY-MM-DD`; the decision date |
| `history` | list | no | The customer's other loans: `loan_id`, `disbursement_date`, `total_amount`. Only loans strictly before `disbursement_date` are used; repeats of a loan id count once |
| `application_id`, `customer_id`, `loan_id` | text | no | Echoed, or used to ignore the application's own loan in `history` |
| `include_features` | true/false | no | Also return the model features (for the decision log and monitoring) |

Unknown fields, out-of-range values and wrong date formats are rejected with HTTP 422 and a message naming the field.

**Response fields**

| Field | Meaning |
| --- | --- |
| `pd` | Probability of default after calibration |
| `pd_raw` | Model output before calibration; log it, recalibration is fitted on it |
| `score` | 300-850, higher is safer; every 20 points doubles the good:bad odds; 600 = 50:1 |
| `band`, `band_label` | A (640+), B (620-639), C (590-619), D (550-589), E (under 550) |
| `decision` | A-B approve; C approve with standard limits; D refer for manual review; E decline |
| `overlay` | `new_market` or `unknown_code` when it overrides the band's decision to "Refer" |
| `reasons` | Up to 3 factors that increased this loan's risk, with their log-odds contribution |
| `history` | How many history entries were used or ignored |
| `model_version` | The model that produced the score |

## Operating the model

| Job | When | Command |
| --- | --- | --- |
| Monitor drift and outcomes | Monthly | `python -m credit_scoring.monitoring decisions.csv --outcomes` |
| Recalibrate the PD level | Mean PD and actual default rate differ by more than 1 point | `python -m credit_scoring.recalibrate matured.csv --from 2024-01-01 --to 2024-06-30` |
| Retrain | AUC more than 0.05 below 0.872, a new product above 5% of volume, or quarterly | `python -m credit_scoring.train` on the extended data |

Thresholds and the full routine are in [MONITORING.md](MONITORING.md). Any change to `artifacts/` reaches Docker only
through a rebuild.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `python` opens the Microsoft Store | Reinstall Python with "Add python.exe to PATH", or use `py -3.12 -m venv .venv` |
| `No matching distribution found for pandas==3.1.0` | Python is older than 3.11: install 3.11-3.13 and recreate `.venv` |
| `import lightgbm` fails with a DLL error | Install the Microsoft Visual C++ Redistributable (x64) |
| `No module named 'credit_scoring'` or `'api'` | Run from the project root: `uvicorn api.main:app`, not `python api/main.py` |
| `FileNotFoundError ... artifacts/model.txt` | `python -m credit_scoring.train` |
| Port 8000 already in use | Stop the other server, or use `--port 8001` / `-p 8080:8000`; find it with `netstat -ano \| findstr :8000` |
| `curl` errors in Windows PowerShell 5 | `curl` is an alias there: use `Invoke-RestMethod` or `curl.exe` |
| Tests show `skipped` | Put `Train.csv` and `Test.csv` in the project root |
| `docker daemon is not running` | Start Docker Desktop and wait for "Engine running" |
| Docker Desktop asks for WSL 2 | In an admin PowerShell: `wsl --install`, then restart |
| Git Bash rewrites `/app/...` paths | Prefix the command with `MSYS_NO_PATHCONV=1`, or use PowerShell |
| Container stays `(unhealthy)` | `docker logs credit-api`; usually missing artifacts: retrain, then rebuild |
