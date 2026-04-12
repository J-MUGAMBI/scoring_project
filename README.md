# Credit Risk Scorer

FastAPI application for credit risk scoring with XGBoost: individual prediction, database-backed batch scoring, data analysis (EDA), model interpretability, and prediction logs. User accounts and history live in **SQLite**.

**Requirements:** Python **3.12** (recommended; match `Dockerfile` and wheels in `requirements.txt`).

---

## What gets created automatically

When the app starts (`python app.py`, `uvicorn`, or Docker), it will:

1. **Create the SQLite database file** (if it does not exist) and all required tables (`users`, `prediction_logs`, `batch_prediction_rows`, `canonical_batch_dataset`, etc.).
2. **Load the canonical batch dataset** into SQLite if the snapshot table is empty: it reads `X_deploy_sample.csv` from the application directory and stores it in `canonical_batch_dataset`. That powers **Batch Scoring** (run from DB), **Data Analysis** (`POST /data_analysis_run` with `{}`), and customer autofill APIs.

You do **not** need a separate migration step for a fresh clone.

---

## Quick start (local)

From this project directory (the folder that contains `app.py`):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 1) Run the web app

**Option A — same as most developers use (reload on code changes):**

```bash
python app.py
```

By default this binds to `127.0.0.1` and port **8765**. Open:

- App: [http://127.0.0.1:8765/](http://127.0.0.1:8765/)
- Swagger UI: [http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs)

**Option B — Uvicorn directly (no reload unless you add `--reload`):**

```bash
uvicorn app:app --host 127.0.0.1 --port 8765 --reload
```

**Listen on all interfaces (e.g. another device on your LAN):**

```bash
set UVICORN_HOST=0.0.0.0
set UVICORN_PORT=8765
python app.py
```

On macOS/Linux use `export UVICORN_HOST=0.0.0.0` instead of `set`.

### 2) Create an account

1. Open `/register` and sign up.
2. Log in at `/login`.

All authenticated pages (batch, individual, data analysis, interpretability, prediction logs) expect a session cookie after login.

### 3) Optional — seed prediction logs from the sample CSV

The **canonical** dataset for batch/EDA is already loaded from `X_deploy_sample.csv` on first startup. To also create **prediction log** entries (same as running batch scoring once), use the helper script **after** you know your user id:

```bash
python bulk_import.py list-users
python bulk_import.py reset-load-sample --user-id 1
```

Or import a specific credit-features CSV (columns must match `feature_list.json`):

```bash
python bulk_import.py credit --csv X_deploy_sample.csv --user-id 1
```

**Raw** spreadsheet import (no model; stores its own table):

```bash
python bulk_import.py raw --csv "your_file.csv"
```

`bulk_import.py` uses the **same database path as the app**: set `SQLITE_DB_PATH` if you override it (see below).

---

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SQLITE_DB_PATH` | Absolute or relative path to the SQLite file | `users.db` next to `app.py` |
| `SESSION_SECRET` | Starlette session signing secret (set in production) | Built-in placeholder (change for deploy) |
| `UVICORN_HOST` | Bind address | `127.0.0.1` when using `python app.py` |
| `UVICORN_PORT` | Port | `8765` |

Ensure the directory for `SQLITE_DB_PATH` exists or is creatable; the app creates the parent directory if needed.

---

## Main routes and pages

| Page | Path | Notes |
|------|------|--------|
| Home | `/` | |
| Register / Login | `/register`, `/login` | |
| Individual scoring | `/individual` | |
| Batch scoring | `/batch` | Uses canonical data in DB |
| Data analysis | `/data_analysis` | Run analysis uses DB (`POST /data_analysis_run` with `{}`) |
| Interpretability | `/interpretability` | |
| Prediction logs | `/prediction_logs` | |

API documentation: `/docs` (log in via `POST /login` in Swagger to use session-protected routes from the docs UI).

---

## Bundled files the app expects

Keep these in the same directory as `app.py` (they are required for predictions and batch features):

- `model.joblib`, `feature_list.json`, `threshold.json`
- `X_deploy_sample.csv` (canonical seed; required until you replace the DB snapshot by other means)

---

## Docker

Build and run from **this directory** (where `docker-compose.yml` lives).

### One-time: set a session secret (recommended)

Create a `.env` file next to `docker-compose.yml`:

```env
SESSION_SECRET=paste-a-long-random-string-here
HOST_PORT=8765
```

Example (Linux/macOS): `openssl rand -hex 32`

### Build and start

```bash
docker compose build
docker compose up -d
```

Open [http://localhost:8765/](http://localhost:8765/) (or the host port you set).

### Useful commands

```bash
docker compose logs -f web
docker compose ps
docker compose down
```

Data is persisted in the named volume `app_data`, mounted at `/app/data` in the container. Compose sets `SQLITE_DB_PATH=/app/data/users.db`, so **user accounts and logs survive** container restarts.

### Fresh database in Docker

Remove the volume and bring the stack up again (this deletes all users and logs in that volume):

```bash
docker compose down -v
docker compose up -d --build
```

Then register a new user. The app will recreate the schema and re-seed `canonical_batch_dataset` from `X_deploy_sample.csv` on startup.

### Importing into the container DB with `bulk_import.py`

Run a one-off container with the same image, env, and volume so the script writes to the same SQLite file:

```bash
docker compose run --rm -e SQLITE_DB_PATH=/app/data/users.db web \
  python bulk_import.py list-users

docker compose run --rm -e SQLITE_DB_PATH=/app/data/users.db web \
  python bulk_import.py reset-load-sample --user-id 1
```

---

## Troubleshooting

- **Port already in use:** Set `UVICORN_PORT` to a free port (e.g. `8770`) or change `HOST_PORT` in Docker.
- **“Authentication required” on pages:** Log in at `/login`. Register first if you have a new database.
- **Batch / data analysis empty:** Confirm `X_deploy_sample.csv` is present and the app was started once so `canonical_batch_dataset` is populated.
- **SciPy / XGBoost install errors on Windows:** Use Python 3.12 and a clean venv; if the host environment still conflicts, run the app with **Docker** (above) for a reproducible Linux stack.

---

## Project layout (high level)

```
.
├── app.py                 # FastAPI app + startup DB init and canonical seed
├── bulk_import.py         # CLI: raw/credit import, list-users, reset-load-sample
├── requirements.txt
├── model.joblib
├── feature_list.json
├── threshold.json
├── X_deploy_sample.csv
├── Dockerfile
├── docker-compose.yml
├── templates/             # Jinja2 UI
└── README.md
```

---

## License

See [LICENSE](LICENSE) if present in the repository.
