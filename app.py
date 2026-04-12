import os
import json
import io
import base64
import random
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import check_password_hash, generate_password_hash

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_db_resolved = os.path.abspath(
    os.environ.get("SQLITE_DB_PATH", os.path.join(BASE_DIR, "users.db"))
)
_db_dir = os.path.dirname(_db_resolved)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)
DB_PATH = _db_resolved

_OPENAPI_DESCRIPTION = (
    "Session-based credit scoring API. **Authenticate first:** open `POST /login` here, execute with your "
    "username/password JSON, then the session cookie is sent on later requests from this docs page.\n\n"
    "**Batch scoring** (`/batch_predict`, `/batch_predict_stored`): JSON field `csv_data` contains only "
    "**CustomerID, Probability, Prediction, Risk_Category, Decision** (no raw application features). "
    "Full rows are still stored server-side for logs.\n\n"
    "**Data analysis without upload:** use **`POST /data_analysis_run`** with an empty body `{}` — it runs "
    "the same Plotly pipeline on the **canonical dataset in SQLite** (no `multipart` file)."
)

app = FastAPI(
    title="Credit Risk Scorer",
    description=_OPENAPI_DESCRIPTION,
    openapi_tags=[
        {
            "name": "Batch scoring",
            "description": "Batch prediction. Response `csv_data` is redacted for API/Swagger consumers.",
        },
        {
            "name": "Data analysis",
            "description": "EDA charts and summary. Prefer `POST /data_analysis_run` (database-backed, no upload).",
        },
        {
            "name": "Individual",
            "description": "Single-customer predict and SHAP-style explain.",
        },
        {
            "name": "Prediction logs",
            "description": "History and batch customer tables (redacted per-row fields).",
        },
        {
            "name": "Auth",
            "description": "Login, logout, registration.",
        },
    ],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "your-secret-key-change-this-in-production"),
    max_age=14 * 24 * 3600,
)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


class DataAnalysisRunBody(BaseModel):
    """Send `{}` from Swagger. The body is ignored; EDA uses the canonical SQLite dataset."""

    model_config = ConfigDict(extra="ignore")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  full_name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS prediction_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  log_type TEXT NOT NULL,
                  probability REAL,
                  prediction INTEGER,
                  risk_category TEXT,
                  decision_label TEXT,
                  threshold REAL,
                  batch_total_records INTEGER,
                  batch_high_risk INTEGER,
                  batch_medium_risk INTEGER,
                  batch_low_risk INTEGER,
                  batch_pred_default INTEGER,
                  batch_pred_no_default INTEGER,
                  input_json TEXT,
                  batch_result_csv TEXT,
                  FOREIGN KEY (user_id) REFERENCES users(id))"""
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_logs_user_created "
            "ON prediction_logs(user_id, created_at DESC)"
        )
        # Existing DBs created before batch_pred_* columns
        for col_sql in (
            "ALTER TABLE prediction_logs ADD COLUMN batch_pred_default INTEGER",
            "ALTER TABLE prediction_logs ADD COLUMN batch_pred_no_default INTEGER",
            "ALTER TABLE prediction_logs ADD COLUMN batch_result_csv TEXT",
        ):
            try:
                c.execute(col_sql)
            except sqlite3.OperationalError:
                pass
        c.execute(
            """CREATE TABLE IF NOT EXISTS batch_prediction_rows (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  prediction_log_id INTEGER NOT NULL,
                  row_index INTEGER NOT NULL,
                  model_class INTEGER NOT NULL,
                  risk_tier TEXT NOT NULL,
                  row_json TEXT NOT NULL,
                  FOREIGN KEY (prediction_log_id) REFERENCES prediction_logs(id) ON DELETE CASCADE
               )"""
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_bpr_log ON batch_prediction_rows(prediction_log_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_bpr_log_row ON batch_prediction_rows(prediction_log_id, row_index)"
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS canonical_batch_dataset (
                  id INTEGER PRIMARY KEY CHECK (id = 1),
                  csv_text TEXT NOT NULL,
                  source_filename TEXT,
                  updated_at TEXT
               )"""
        )
        conn.commit()
    finally:
        conn.close()


CUSTOMER_ID_COL = "CustomerID"


def _unique_six_digit_customer_ids(n: int) -> List[int]:
    """Random unique integers in [100000, 999999] (six digits)."""
    if n <= 0:
        return []
    if n > 900_000:
        raise ValueError("Too many rows for unique 6-digit CustomerIDs")
    return random.sample(range(100_000, 1_000_000), n)


def ensure_customer_id_first_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure CustomerID exists as the first column; generate unique 6-digit IDs if missing."""
    if df.empty:
        return df.copy()
    out = df.copy()
    n = len(out)
    if CUSTOMER_ID_COL in out.columns:
        rest = [c for c in out.columns if c != CUSTOMER_ID_COL]
        return out[[CUSTOMER_ID_COL] + rest]
    ids = _unique_six_digit_customer_ids(n)
    first = pd.DataFrame({CUSTOMER_ID_COL: ids})
    for c in out.columns:
        first[c] = out[c].values
    return first


def _canonical_snapshot_needs_customer_id(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    if CUSTOMER_ID_COL not in df.columns:
        return True
    return df.columns[0] != CUSTOMER_ID_COL


def persist_canonical_snapshot(df: pd.DataFrame, source_filename: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    text = buf.getvalue()
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            """INSERT OR REPLACE INTO canonical_batch_dataset (id, csv_text, source_filename, updated_at)
               VALUES (1, ?, ?, ?)""",
            (text, source_filename, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def ensure_canonical_batch_dataset() -> None:
    """Seed the canonical batch CSV from disk if the DB snapshot is empty."""
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT LENGTH(csv_text) FROM canonical_batch_dataset WHERE id = 1")
        row = c.fetchone()
        if row and row[0] and row[0] > 20:
            return
        path = os.path.join(BASE_DIR, "X_deploy_sample.csv")
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        df = pd.read_csv(io.StringIO(raw))
        df = ensure_customer_id_first_column(df)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        text = buf.getvalue()
        c.execute(
            """INSERT OR REPLACE INTO canonical_batch_dataset (id, csv_text, source_filename, updated_at)
               VALUES (1, ?, ?, ?)""",
            (text, "X_deploy_sample.csv", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def sync_canonical_customer_ids() -> None:
    """Migrate stored canonical CSV: add CustomerID as first column with unique 6-digit values."""
    ensure_canonical_batch_dataset()
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT csv_text, source_filename FROM canonical_batch_dataset WHERE id = 1")
        r = c.fetchone()
        if not r or not r[0]:
            return
        df = pd.read_csv(io.StringIO(r[0]))
        fn = r[1] or "stored_dataset.csv"
        if not _canonical_snapshot_needs_customer_id(df):
            return
        persist_canonical_snapshot(ensure_customer_id_first_column(df), fn)
    finally:
        conn.close()


def get_canonical_batch_status() -> Dict[str, Any]:
    """Row count and label for the stored batch dataset (used by the batch UI)."""
    sync_canonical_customer_ids()
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT csv_text, source_filename FROM canonical_batch_dataset WHERE id = 1")
        r = c.fetchone()
        if not r or not r[0]:
            return {
                "loaded": False,
                "row_count": 0,
                "source_filename": None,
                "message": "No dataset in database. Add X_deploy_sample.csv next to the app.",
            }
        df = pd.read_csv(io.StringIO(r[0]))
        return {
            "loaded": True,
            "row_count": len(df),
            "source_filename": r[1] or "stored_dataset.csv",
        }
    finally:
        conn.close()


def get_canonical_batch_df_and_label() -> Tuple[pd.DataFrame, str]:
    sync_canonical_customer_ids()
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT csv_text, source_filename FROM canonical_batch_dataset WHERE id = 1")
        r = c.fetchone()
        if not r or not r[0]:
            raise ValueError(
                "No batch dataset is available. Ensure X_deploy_sample.csv exists in the "
                "application directory or load data into the database."
            )
        label = r[1] or "stored_dataset.csv"
        return pd.read_csv(io.StringIO(r[0])), label
    finally:
        conn.close()


init_db()
ensure_canonical_batch_dataset()
sync_canonical_customer_ids()

model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
with open(os.path.join(BASE_DIR, "feature_list.json"), "r", encoding="utf-8") as f:
    features: List[str] = json.load(f)
with open(os.path.join(BASE_DIR, "threshold.json"), "r", encoding="utf-8") as f:
    threshold_config = json.load(f)
    BEST_THRESHOLD = threshold_config["best_threshold"]

REQUIRED_ANALYSIS_COLUMNS = [
    "CRR_NARRATION",
    "Max_REMAINING_TENOR(AllLoans)",
    "RUNNING_LOANS_COUNT",
    "Exposure_Amount",
    "NetIncome",
    "OnUsEMI",
    "DrTurnover",
    "TotalAssets",
    "MobileTotal",
    "MaxArrears",
    "CurrentArrears",
    "NonPerforming",
    "Gender",
    "SavingAcctDepositCount",
    "Age",
    "EmployerStrength",
    "SECTOR",
    "CUSTOMER_SUBSEGMENT_NAME",
    "NATIONALITY",
    "EMPLOYEMENT_STATUS",
    "MARITAL_STATUS",
    "AML_RISK_CLASS",
    "CustomerTenure",
]


def require_login(request: Request) -> None:
    if request.session.get("user_id") is None:
        raise HTTPException(status_code=401, detail="Authentication required")


def require_login_redirect(request: Request) -> Optional[RedirectResponse]:
    if request.session.get("user_id") is None:
        return RedirectResponse(url="/login", status_code=302)
    return None


INPUT_JSON_MAX_CHARS = 2_000_000
# Store full scored CSV for batch logs; rebuild tables on read (avoids JSON truncation).
BATCH_RESULT_CSV_MAX_CHARS = 12_000_000


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def build_batch_customer_tables(
    results_df: pd.DataFrame,
    predictions: np.ndarray,
    risk_categories: List[str],
    max_rows_per_group: int,
) -> Dict[str, Any]:
    """Split batch results into row dicts by model class (0/1) and by policy risk tier."""
    df = results_df.copy().reset_index(drop=True)
    pred_arr = np.asarray(predictions).astype(int).flatten()
    rc_arr = np.asarray(risk_categories, dtype=object).flatten()
    m = min(len(df), len(pred_arr), len(rc_arr))
    df = df.iloc[:m].reset_index(drop=True)
    pred_arr = pred_arr[:m]
    rc_arr = rc_arr[:m]
    df.insert(0, "batch_row_index", np.arange(1, len(df) + 1))

    def slice_by_indices(indices: np.ndarray) -> tuple[List[Dict[str, Any]], int]:
        total = int(len(indices))
        if total == 0:
            return [], 0
        take = indices[:max_rows_per_group]
        sub = df.iloc[take]
        return _df_to_records(sub), total

    idx_def = np.flatnonzero(pred_arr == 1)
    idx_ok = np.flatnonzero(pred_arr == 0)
    r_def, t_def = slice_by_indices(idx_def)
    r_ok, t_ok = slice_by_indices(idx_ok)
    by_mc = {
        "default_risk": {"rows": r_def, "total_in_group": t_def, "shown": len(r_def)},
        "no_default": {"rows": r_ok, "total_in_group": t_ok, "shown": len(r_ok)},
    }

    by_rt: Dict[str, Any] = {}
    for label in ("HIGH RISK", "MEDIUM RISK", "LOW RISK"):
        tier_idx = np.flatnonzero(np.asarray(rc_arr, dtype=str) == label)
        recs, tot = slice_by_indices(tier_idx)
        by_rt[label] = {"rows": recs, "total_in_group": tot, "shown": len(recs)}

    any_trunc = any(
        g["total_in_group"] > g["shown"]
        for g in list(by_mc.values()) + list(by_rt.values())
    )
    return {
        "by_model_class": by_mc,
        "by_risk_tier": by_rt,
        "truncated": any_trunc,
        "max_rows_per_section": max_rows_per_group,
    }


def customer_tables_from_result_csv(
    csv_text: str, max_rows_per_group: int = 200
) -> Dict[str, Any]:
    """Rebuild grouped tables from stored batch result CSV (same columns as batch download)."""
    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty:
        return {"by_model_class": {}, "by_risk_tier": {}, "error": "empty_csv"}
    if "Prediction" not in df.columns or "Risk_Category" not in df.columns:
        return {
            "by_model_class": {},
            "by_risk_tier": {},
            "error": "missing_prediction_columns",
        }
    pred = pd.to_numeric(df["Prediction"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    rc = df["Risk_Category"].astype(str).tolist()
    return build_batch_customer_tables(df, pred, rc, max_rows_per_group)


BATCH_ROW_TABLE_MAX = 100_000
DISPLAY_ROWS_PER_GROUP = 200


def customer_tables_from_stored_rows(
    conn: sqlite3.Connection, log_id: int, max_per_group: int = DISPLAY_ROWS_PER_GROUP
) -> Optional[Dict[str, Any]]:
    """Build the same structure as build_batch_customer_tables from normalized DB rows."""
    c = conn.cursor()
    c.execute(
        """SELECT row_index, model_class, risk_tier, row_json
           FROM batch_prediction_rows WHERE prediction_log_id = ?
           ORDER BY row_index""",
        (log_id,),
    )
    fetched = c.fetchall()
    if not fetched:
        return None
    rec_def: List[Dict[str, Any]] = []
    rec_ok: List[Dict[str, Any]] = []
    rec_h: List[Dict[str, Any]] = []
    rec_m: List[Dict[str, Any]] = []
    rec_l: List[Dict[str, Any]] = []
    for _ri, mc, tier, rj in fetched:
        obj = json.loads(rj)
        if int(mc) == 1:
            rec_def.append(obj)
        else:
            rec_ok.append(obj)
        if tier == "HIGH RISK":
            rec_h.append(obj)
        elif tier == "MEDIUM RISK":
            rec_m.append(obj)
        elif tier == "LOW RISK":
            rec_l.append(obj)

    def cap(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(recs)
        take = recs[:max_per_group]
        return {"rows": take, "total_in_group": total, "shown": len(take)}

    by_mc = {"default_risk": cap(rec_def), "no_default": cap(rec_ok)}
    by_rt = {
        "HIGH RISK": cap(rec_h),
        "MEDIUM RISK": cap(rec_m),
        "LOW RISK": cap(rec_l),
    }
    any_trunc = any(
        g["total_in_group"] > g["shown"] for g in list(by_mc.values()) + list(by_rt.values())
    )
    return {
        "by_model_class": by_mc,
        "by_risk_tier": by_rt,
        "truncated": any_trunc,
        "max_rows_per_section": max_per_group,
    }


BATCH_LOG_TABLE_KEYS: Tuple[str, ...] = (
    "CustomerID",
    "Probability",
    "Prediction",
    "Risk_Category",
    "Decision",
)


def scored_results_to_api_safe_csv(results_df: pd.DataFrame) -> str:
    """CSV for API/JSON responses: only CustomerID + model outputs (no credit feature columns)."""
    df = results_df.reset_index(drop=True)
    n = len(df)
    parts: Dict[str, Any] = {}
    for k in BATCH_LOG_TABLE_KEYS:
        parts[k] = df[k].copy() if k in df.columns else pd.Series([np.nan] * n, dtype=object)
    safe = pd.DataFrame(parts)
    for i in range(n):
        if pd.isna(safe.at[i, "CustomerID"]):
            safe.at[i, "CustomerID"] = i + 1
    buf = io.StringIO()
    safe.to_csv(buf, index=False)
    return buf.getvalue()


def redact_batch_log_customer_tables(customer_tables: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only non-sensitive columns for the prediction logs UI."""

    def one_row(row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {k: row.get(k) for k in BATCH_LOG_TABLE_KEYS}
        if out.get("CustomerID") is None:
            bri = row.get("batch_row_index")
            if bri is not None:
                out["CustomerID"] = bri
        return out

    def cap_group(group: Any) -> Any:
        if not isinstance(group, dict):
            return group
        rows = group.get("rows")
        if not isinstance(rows, list):
            return group
        return {**group, "rows": [one_row(r) for r in rows if isinstance(r, dict)]}

    out = dict(customer_tables)
    for key in ("by_model_class", "by_risk_tier"):
        section = out.get(key)
        if not isinstance(section, dict):
            continue
        out[key] = {name: cap_group(g) for name, g in section.items()}
    return out


def insert_batch_prediction_log_transaction(
    user_id: int,
    *,
    threshold: float,
    batch_total: int,
    batch_high: int,
    batch_med: int,
    batch_low: int,
    batch_pred_default: int,
    batch_pred_no_default: int,
    input_payload: Dict[str, Any],
    results_df: pd.DataFrame,
    predictions: np.ndarray,
    risk_categories: List[str],
) -> Optional[int]:
    """Insert batch log header + one row per account. Returns prediction_logs.id or None on failure."""
    df = results_df.reset_index(drop=True)
    pred_a = np.asarray(predictions).astype(int).flatten()
    rc_a = np.asarray(risk_categories, dtype=object).flatten()
    m = min(len(df), len(pred_a), len(rc_a))
    if m > BATCH_ROW_TABLE_MAX:
        return None
    input_json = json.dumps(input_payload, default=str)
    if len(input_json) > INPUT_JSON_MAX_CHARS:
        input_json = input_json[:INPUT_JSON_MAX_CHARS] + "..."
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        c = conn.cursor()
        c.execute(
            """INSERT INTO prediction_logs (
                user_id, log_type, probability, prediction, risk_category, decision_label,
                threshold, batch_total_records, batch_high_risk, batch_medium_risk, batch_low_risk,
                batch_pred_default, batch_pred_no_default, input_json, batch_result_csv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                "batch",
                None,
                None,
                None,
                None,
                threshold,
                batch_total,
                batch_high,
                batch_med,
                batch_low,
                batch_pred_default,
                batch_pred_no_default,
                input_json,
                None,
            ),
        )
        log_id = int(c.lastrowid)
        buf: List[tuple] = []
        for i in range(m):
            one = json.loads(
                df.iloc[[i]].to_json(orient="records", date_format="iso", default_handler=str)
            )[0]
            one["batch_row_index"] = i + 1
            rj = json.dumps(one, default=str)
            buf.append(
                (
                    log_id,
                    i + 1,
                    int(pred_a[i]),
                    str(rc_a[i]),
                    rj,
                )
            )
        c.executemany(
            """INSERT INTO batch_prediction_rows
               (prediction_log_id, row_index, model_class, risk_tier, row_json)
               VALUES (?, ?, ?, ?, ?)""",
            buf,
        )
        conn.commit()
        return log_id
    except Exception:
        conn.rollback()
        return None
    finally:
        conn.close()


def run_batch_scoring_core(
    df: pd.DataFrame,
    filename_for_log: str,
    user_id: int,
    *,
    input_extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a batch DataFrame and persist log. Raises ValueError or RuntimeError."""
    df = ensure_customer_id_first_column(df)
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    input_data = df[features]
    probabilities = model.predict_proba(input_data)[:, 1]
    predictions = (probabilities >= BEST_THRESHOLD).astype(int)
    decisions = [decision_from_pd(p) for p in probabilities]
    risk_categories = [d["Risk_Class"] for d in decisions]
    decision_labels = [d["Decision"] for d in decisions]
    results_df = df.copy()
    results_df["Probability"] = probabilities
    results_df["Prediction"] = predictions
    results_df["Risk_Category"] = risk_categories
    results_df["Decision"] = decision_labels
    high_risk = sum(1 for r in risk_categories if r == "HIGH RISK")
    medium_risk = sum(1 for r in risk_categories if r == "MEDIUM RISK")
    low_risk = sum(1 for r in risk_categories if r == "LOW RISK")
    n_pred_default = int(predictions.sum())
    n_pred_no_default = int(len(predictions) - n_pred_default)
    input_payload: Dict[str, Any] = {
        "filename": filename_for_log,
        "rows": len(df),
        "threshold": BEST_THRESHOLD,
        "predicted_default_risk_rows": n_pred_default,
        "predicted_no_default_rows": n_pred_no_default,
        "risk_tiers": {
            "HIGH RISK": high_risk,
            "MEDIUM RISK": medium_risk,
            "LOW RISK": low_risk,
        },
        "predictions_in_database": True,
    }
    if input_extras:
        input_payload.update(input_extras)
    log_id = insert_batch_prediction_log_transaction(
        user_id,
        threshold=BEST_THRESHOLD,
        batch_total=len(df),
        batch_high=high_risk,
        batch_med=medium_risk,
        batch_low=low_risk,
        batch_pred_default=n_pred_default,
        batch_pred_no_default=n_pred_no_default,
        input_payload=input_payload,
        results_df=results_df,
        predictions=predictions,
        risk_categories=risk_categories,
    )
    if log_id is None:
        raise RuntimeError("Could not save batch predictions to the database.")
    return {
        "success": True,
        "total_records": len(df),
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "csv_data": scored_results_to_api_safe_csv(results_df),
    }


def replace_batch_rows_from_result_csv(conn: sqlite3.Connection, log_id: int, csv_text: str) -> None:
    """Replace row predictions for a log from scored CSV (used by attach + migration)."""
    df = pd.read_csv(io.StringIO(csv_text))
    c = conn.cursor()
    c.execute("DELETE FROM batch_prediction_rows WHERE prediction_log_id = ?", (log_id,))
    if df.empty:
        return
    if "Prediction" not in df.columns or "Risk_Category" not in df.columns:
        return
    pred = pd.to_numeric(df["Prediction"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    rc = df["Risk_Category"].astype(str).to_numpy()
    m = min(len(df), len(pred), len(rc))
    for i in range(m):
        one = json.loads(
            df.iloc[[i]].to_json(orient="records", date_format="iso", default_handler=str)
        )[0]
        one["batch_row_index"] = i + 1
        rj = json.dumps(one, default=str)
        c.execute(
            """INSERT INTO batch_prediction_rows
               (prediction_log_id, row_index, model_class, risk_tier, row_json)
               VALUES (?, ?, ?, ?, ?)""",
            (log_id, i + 1, int(pred[i]), str(rc[i]), rj),
        )


def insert_prediction_log(
    user_id: int,
    log_type: str,
    *,
    probability: Optional[float] = None,
    prediction: Optional[int] = None,
    risk_category: Optional[str] = None,
    decision_label: Optional[str] = None,
    threshold: Optional[float] = None,
    batch_total: Optional[int] = None,
    batch_high: Optional[int] = None,
    batch_med: Optional[int] = None,
    batch_low: Optional[int] = None,
    batch_pred_default: Optional[int] = None,
    batch_pred_no_default: Optional[int] = None,
    input_payload: Optional[Dict[str, Any]] = None,
    batch_result_csv: Optional[str] = None,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        input_json: Optional[str] = None
        if input_payload is not None:
            input_json = json.dumps(input_payload, default=str)
            if len(input_json) > INPUT_JSON_MAX_CHARS:
                input_json = input_json[:INPUT_JSON_MAX_CHARS] + "..."
        c.execute(
            """INSERT INTO prediction_logs (
                user_id, log_type, probability, prediction, risk_category, decision_label,
                threshold, batch_total_records, batch_high_risk, batch_medium_risk, batch_low_risk,
                batch_pred_default, batch_pred_no_default, input_json, batch_result_csv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                log_type,
                probability,
                prediction,
                risk_category,
                decision_label,
                threshold,
                batch_total,
                batch_high,
                batch_med,
                batch_low,
                batch_pred_default,
                batch_pred_no_default,
                input_json,
                batch_result_csv,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def safe_insert_prediction_log(*args: Any, **kwargs: Any) -> None:
    try:
        insert_prediction_log(*args, **kwargs)
    except Exception:
        pass


def fig_to_json_dict(fig: go.Figure) -> dict:
    return json.loads(pio.to_json(fig))


def decision_from_pd(pd_val: float) -> Dict[str, str]:
    if pd_val > 0.10:
        return {"Risk_Class": "HIGH RISK", "Decision": "DECLINE"}
    if pd_val > 0.03:
        return {
            "Risk_Class": "MEDIUM RISK",
            "Decision": "REFER / APPROVE WITH CONTROLS",
        }
    return {"Risk_Class": "LOW RISK", "Decision": "AUTO APPROVE"}


def prepare_input_data(data_dict: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    return df[features]


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html")


@app.post("/register")
async def register(data: Dict[str, Any] = Body(...)):
    try:
        full_name = data.get("fullName")
        email = data.get("email")
        username = data.get("username")
        password = data.get("password")
        if not all([full_name, email, username, password]):
            return JSONResponse({"error": "All fields are required"}, status_code=400)
        if len(password) < 8:
            return JSONResponse(
                {"error": "Password must be at least 8 characters"}, status_code=400
            )
        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (full_name, email, username, password) VALUES (?, ?, ?, ?)",
                (full_name, email, username, hashed_password),
            )
            conn.commit()
            return {"success": True, "message": "Registration successful"}
        except sqlite3.IntegrityError:
            return JSONResponse(
                {"error": "Username or email already exists"}, status_code=400
            )
        finally:
            conn.close()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.post("/login", tags=["Auth"], summary="JSON login (sets session cookie for Swagger)")
async def login(request: Request, data: Dict[str, Any] = Body(...)):
    try:
        username = data.get("username")
        password = data.get("password")
        if not all([username, password]):
            return JSONResponse(
                {"error": "Username and password are required"}, status_code=400
            )
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT id, username, password, full_name FROM users WHERE username = ? OR email = ?",
            (username, username),
        )
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            request.session["user_id"] = user[0]
            request.session["username"] = user[1]
            request.session["full_name"] = user[3]
            return {"success": True, "message": "Login successful"}
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    redir = require_login_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(
        request,
        "index.html",
        {"user": request.session.get("full_name")},
    )


@app.get("/individual", response_class=HTMLResponse)
async def individual_scoring(request: Request):
    redir = require_login_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(
        request,
        "individual.html",
        {"features": features},
    )


@app.post("/predict", tags=["Individual"], summary="Score one customer (JSON body)")
async def predict(request: Request, data: Dict[str, Any] = Body(...)):
    require_login(request)
    try:
        input_df = prepare_input_data(data)
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = int(probability >= BEST_THRESHOLD)
        decision_info = decision_from_pd(probability)
        uid = request.session["user_id"]
        safe_insert_prediction_log(
            uid,
            "individual",
            probability=probability,
            prediction=prediction,
            risk_category=decision_info["Risk_Class"],
            decision_label=decision_info["Decision"],
            threshold=BEST_THRESHOLD,
            input_payload=data,
        )
        return {
            "probability": probability,
            "prediction": prediction,
            "risk_category": decision_info["Risk_Class"],
            "decision": decision_info["Decision"],
            "threshold": BEST_THRESHOLD,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/batch", response_class=HTMLResponse)
async def batch_scoring(request: Request):
    redir = require_login_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "batch.html")


@app.get(
    "/api/batch_dataset_status",
    tags=["Batch scoring"],
    summary="Canonical batch dataset row count (session)",
)
async def api_batch_dataset_status(request: Request):
    require_login(request)
    return get_canonical_batch_status()


@app.post(
    "/batch_predict_stored",
    tags=["Batch scoring"],
    summary="Score canonical DB snapshot",
    description=(
        "Returns counts and **redacted** `csv_data` (CustomerID, Probability, Prediction, "
        "Risk_Category, Decision only). Full feature rows are persisted internally for logs."
    ),
)
async def batch_predict_stored(request: Request):
    require_login(request)
    try:
        df, label = get_canonical_batch_df_and_label()
        uid = request.session["user_id"]
        return run_batch_scoring_core(
            df,
            label,
            uid,
            input_extras={"data_source": "canonical_database"},
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post(
    "/batch_predict",
    tags=["Batch scoring"],
    summary="Score uploaded CSV",
    description="Multipart file upload. Response `csv_data` is redacted like batch_predict_stored.",
)
async def batch_predict(request: Request, file: UploadFile = File(...)):
    require_login(request)
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        uid = request.session["user_id"]
        return run_batch_scoring_core(df, file.filename or "upload.csv", uid)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def run_data_analysis_from_dataframe(df_in: pd.DataFrame) -> Dict[str, Any]:
    """Shared EDA pipeline for uploaded CSV or canonical DB dataset."""
    missing_columns = [col for col in REQUIRED_ANALYSIS_COLUMNS if col not in df_in.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    df = df_in[REQUIRED_ANALYSIS_COLUMNS].copy()
    if len(df) > 5000:
        df_sample = df.sample(5000, random_state=42)
    else:
        df_sample = df
    numeric_df = df_sample.select_dtypes(include=[np.number])
    summary_table_html = numeric_df.describe().transpose().to_html(
        classes="table table-striped table-sm table-bordered", border=0
    )
    plots: Dict[str, Any] = {}
    if "Gender" in df_sample.columns:
        gender_counts = df_sample["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig_gender = px.pie(
            gender_counts,
            names="Gender",
            values="Count",
            title="<b>Gender Distribution</b>",
        )
        fig_gender.update_traces(textposition="inside", textinfo="percent+label")
        plots["gender_pie"] = fig_to_json_dict(fig_gender)
    if "SECTOR" in df_sample.columns:
        sector_counts = df_sample["SECTOR"].value_counts().nlargest(10).reset_index()
        sector_counts.columns = ["SECTOR", "Count"]
        fig_sector = px.bar(
            sector_counts,
            x="SECTOR",
            y="Count",
            title="<b>Customers by Sector</b>",
        )
        fig_sector.update_layout(
            xaxis_title="<b>Sector</b>", yaxis_title="<b>Number of Customers</b>"
        )
        plots["sector_bar"] = fig_to_json_dict(fig_sector)
    if "NetIncome" in df_sample.columns and "SECTOR" in df_sample.columns:
        income_sector = df_sample.groupby("SECTOR", as_index=False)["NetIncome"].mean()
        fig_income = px.bar(
            income_sector,
            x="SECTOR",
            y="NetIncome",
            title="<b>Average Net Income by Sector</b>",
        )
        fig_income.update_layout(
            xaxis_title="<b>Sector</b>", yaxis_title="<b>Average Net Income</b>"
        )
        plots["income_sector_bar"] = fig_to_json_dict(fig_income)
    if "Gender" in df_sample.columns and "NonPerforming" in df_sample.columns:
        nonperf_gender = (
            df_sample.groupby(["Gender", "NonPerforming"]).size().reset_index(name="Count")
        )
        fig_nonperf = px.bar(
            nonperf_gender,
            x="Gender",
            y="Count",
            color="NonPerforming",
            barmode="stack",
            title="<b>Non-Performing Loans by Gender</b>",
        )
        fig_nonperf.update_layout(
            xaxis_title="<b>Gender</b>",
            yaxis_title="<b>Number of Accounts</b>",
            legend_title="<b>Non-Performing</b>",
        )
        plots["nonperforming_gender_stacked"] = fig_to_json_dict(fig_nonperf)
    if "Age" in df_sample.columns:
        age_series = df_sample["Age"].dropna()
        if not age_series.empty:
            age_bins = pd.cut(age_series, bins=5)
            age_counts = age_bins.value_counts().sort_index().reset_index()
            age_counts.columns = ["AgeRange", "Count"]
            age_counts["AgeRange"] = age_counts["AgeRange"].astype(str)
            fig_age_bins = px.bar(
                age_counts,
                x="AgeRange",
                y="Count",
                title="<b>Age Distribution (5 Bins)</b>",
            )
            fig_age_bins.update_layout(
                xaxis_title="<b>Age Range</b>", yaxis_title="<b>Number of Customers</b>"
            )
            plots["age_bins"] = fig_to_json_dict(fig_age_bins)
    if "AML_RISK_CLASS" in df_sample.columns:
        aml_counts = df_sample["AML_RISK_CLASS"].value_counts().reset_index()
        aml_counts.columns = ["AML_RISK_CLASS", "Count"]
        fig_aml = px.pie(
            aml_counts,
            names="AML_RISK_CLASS",
            values="Count",
            title="<b>AML Risk Class Distribution</b>",
        )
        fig_aml.update_traces(textposition="inside", textinfo="percent+label")
        plots["aml_risk_pie"] = fig_to_json_dict(fig_aml)
    return {
        "success": True,
        "summary_table": summary_table_html,
        "plots": plots,
    }


@app.get("/data_analysis", response_class=HTMLResponse)
async def data_analysis(request: Request):
    redir = require_login_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "data_analysis.html")


@app.get(
    "/api/canonical_dataset/customers",
    tags=["Batch scoring"],
    summary="List customers + features for autofill UIs (session)",
)
async def api_canonical_dataset_customers(request: Request):
    """Rows from the stored canonical dataset for individual scoring autofill."""
    require_login(request)
    try:
        df, _ = get_canonical_batch_df_and_label()
    except ValueError as e:
        return {"customers": [], "error": str(e)}
    if CUSTOMER_ID_COL not in df.columns:
        return {
            "customers": [],
            "error": "Canonical dataset has no CustomerID column. Reseed the batch dataset.",
        }
    feat_cols = [c for c in features if c in df.columns]
    cols = [CUSTOMER_ID_COL] + feat_cols
    sub = df[cols].copy()
    sub["_sort_key"] = pd.to_numeric(sub[CUSTOMER_ID_COL], errors="coerce")
    sub = sub.sort_values("_sort_key", na_position="last").drop(columns=["_sort_key"])
    records: List[Dict[str, Any]] = []
    for _, row in sub.iterrows():
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = None
            elif hasattr(v, "item"):
                rec[k] = v.item()
            else:
                rec[k] = v
        records.append(rec)
    return {"customers": records}


@app.post(
    "/data_analysis_run",
    tags=["Data analysis"],
    summary="Run EDA on SQLite canonical dataset (no file)",
    description=(
        "Request body must be JSON **`{}`**. Does not accept multipart upload. "
        "Same Plotly + summary pipeline as `data_analysis_upload`."
    ),
)
async def data_analysis_run(
    request: Request,
    _body: DataAnalysisRunBody = Body(
        ...,
        openapi_examples={
            "empty": {
                "summary": "Empty object",
                "value": {},
            }
        },
    ),
):
    require_login(request)
    try:
        df, _ = get_canonical_batch_df_and_label()
        return run_data_analysis_from_dataframe(df)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post(
    "/data_analysis_upload",
    tags=["Data analysis"],
    summary="Run EDA on uploaded CSV (multipart)",
    description="Optional alternative to `data_analysis_run` when you want a custom file.",
)
async def data_analysis_upload(request: Request, file: UploadFile = File(...)):
    require_login(request)
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        return run_data_analysis_from_dataframe(df)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/interpretability", response_class=HTMLResponse)
async def interpretability(request: Request):
    redir = require_login_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(
        request,
        "interpretability.html",
        {"features": features},
    )


@app.post("/explain", tags=["Individual"], summary="Explain prediction (feature plot + tips)")
async def explain_prediction(request: Request, data: Dict[str, Any] = Body(...)):
    require_login(request)
    try:
        input_df = prepare_input_data(data)
        probability = float(model.predict_proba(input_df)[0][1])
        decision_info = decision_from_pd(probability)
        risk_category = decision_info["Risk_Class"]
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
        elif hasattr(model, "named_steps"):
            final_estimator = list(model.named_steps.values())[-1]
            if hasattr(final_estimator, "feature_importances_"):
                feature_importance = final_estimator.feature_importances_
            else:
                feature_importance = np.ones(len(features)) / len(features)
        else:
            feature_importance = np.ones(len(features)) / len(features)
        importance_dict = dict(zip(features, feature_importance))
        sorted_importance = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        top_features = [(f[0], float(f[1])) for f in sorted_importance[:10]]
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_names)), importances)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Feature Importance")
        plt.title("Top 10 Most Important Features")
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        recommendations = generate_recommendations(data, risk_category, top_features)
        uid = request.session["user_id"]
        pred_bin = int(probability >= BEST_THRESHOLD)
        safe_insert_prediction_log(
            uid,
            "explain",
            probability=probability,
            prediction=pred_bin,
            risk_category=risk_category,
            decision_label=decision_info["Decision"],
            threshold=BEST_THRESHOLD,
            input_payload=data,
        )
        return {
            "probability": probability,
            "risk_category": risk_category,
            "top_features": top_features,
            "feature_plot": img_str,
            "recommendations": recommendations,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/prediction_logs", response_class=HTMLResponse)
async def prediction_logs_page(request: Request):
    redir = require_login_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(
        request,
        "prediction_logs.html",
        {"user": request.session.get("full_name")},
    )


@app.get(
    "/api/prediction_logs",
    tags=["Prediction logs"],
    summary="List prediction history (session)",
)
async def api_prediction_logs(request: Request, limit: int = 100, offset: int = 0):
    require_login(request)
    uid = request.session["user_id"]
    limit = min(max(limit, 1), 500)
    offset = max(offset, 0)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        c = conn.cursor()
        c.execute(
            """SELECT id, created_at, log_type, probability, prediction, risk_category,
                      decision_label, threshold, batch_total_records, batch_high_risk,
                      batch_medium_risk, batch_low_risk, batch_pred_default, batch_pred_no_default,
                      CASE WHEN input_json IS NULL THEN NULL
                           WHEN length(input_json) > 500
                           THEN substr(input_json, 1, 500) || '...'
                           ELSE input_json END AS input_preview
               FROM prediction_logs WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (uid, limit, offset),
        )
        rows = [dict(r) for r in c.fetchall()]
        c.execute(
            "SELECT COUNT(*) FROM prediction_logs WHERE user_id = ?",
            (uid,),
        )
        total = c.fetchone()[0]
        return {"logs": rows, "total": total, "limit": limit, "offset": offset}
    finally:
        conn.close()


@app.get(
    "/api/prediction_logs/{log_id}",
    tags=["Prediction logs"],
    summary="Log detail; batch customer_tables rows are redacted",
)
async def api_prediction_log_detail(request: Request, log_id: int):
    """Full log row including parsed `input_json` (for batch customer tables)."""
    require_login(request)
    uid = request.session["user_id"]
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        c = conn.cursor()
        c.execute(
            """SELECT id, created_at, log_type, probability, prediction, risk_category,
                      decision_label, threshold, batch_total_records, batch_high_risk,
                      batch_medium_risk, batch_low_risk, batch_pred_default, batch_pred_no_default,
                      input_json, batch_result_csv
               FROM prediction_logs WHERE id = ? AND user_id = ?""",
            (log_id, uid),
        )
        row = c.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Log not found")
        out = dict(row)
        batch_csv = out.pop("batch_result_csv", None)
        raw = out.pop("input_json", None)
        parsed: Optional[Dict[str, Any]] = None
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
                out["input_parse_error"] = True
        if parsed is None:
            parsed = {}
        pid = out.get("id")
        c.execute(
            "SELECT 1 FROM batch_prediction_rows WHERE prediction_log_id = ? LIMIT 1",
            (pid,),
        )
        has_db_rows = c.fetchone() is not None
        if out.get("log_type") == "batch" and has_db_rows:
            tbl = customer_tables_from_stored_rows(conn, int(pid))
            if tbl:
                parsed["customer_tables"] = tbl
                parsed["customer_tables_source"] = "database"
        elif out.get("log_type") == "batch" and batch_csv:
            parsed["customer_tables"] = customer_tables_from_result_csv(batch_csv)
            parsed["customer_tables_source"] = "stored_result_csv"
        elif out.get("log_type") == "batch" and not has_db_rows and not batch_csv:
            ct = parsed.get("customer_tables") or {}
            if not ct.get("by_model_class"):
                parsed["customer_tables"] = {
                    "omitted": True,
                    "reason": (
                        "This batch log predates per-row storage. Run Batch scoring once more "
                        "to create a new entry with predictions loaded automatically from the database."
                    ),
                }
        ct_final = parsed.get("customer_tables")
        if isinstance(ct_final, dict) and not ct_final.get("omitted") and not ct_final.get("error"):
            parsed["customer_tables"] = redact_batch_log_customer_tables(ct_final)
        out["input_parsed"] = parsed
        return out
    finally:
        conn.close()


@app.post(
    "/api/prediction_logs/{log_id}/attach_result_csv",
    tags=["Prediction logs"],
    summary="Attach full scored CSV to a batch log (multipart)",
)
async def attach_batch_result_csv(
    request: Request, log_id: int, file: UploadFile = File(...)
):
    """Attach the scored batch CSV (same as Batch download) to fill in customer tables for older logs."""
    require_login(request)
    uid = request.session["user_id"]
    raw = await file.read()
    if len(raw) > BATCH_RESULT_CSV_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {BATCH_RESULT_CSV_MAX_CHARS // 1_000_000}MB of text).",
        )
    text = raw.decode("utf-8", errors="replace")
    tables = customer_tables_from_result_csv(text)
    err = tables.get("error")
    if err == "empty_csv":
        raise HTTPException(status_code=400, detail="CSV is empty.")
    if err == "missing_prediction_columns":
        raise HTTPException(
            status_code=400,
            detail="CSV must include Prediction and Risk_Category columns (use the Batch download file).",
        )

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        c = conn.cursor()
        c.execute(
            """UPDATE prediction_logs SET batch_result_csv = ?
               WHERE id = ? AND user_id = ? AND log_type = 'batch'""",
            (text, log_id, uid),
        )
        if c.rowcount == 0:
            raise HTTPException(status_code=404, detail="Batch log not found")
        replace_batch_rows_from_result_csv(conn, log_id, text)
        c.execute("SELECT input_json FROM prediction_logs WHERE id = ?", (log_id,))
        row = c.fetchone()
        if row and row[0]:
            try:
                meta = json.loads(row[0])
                meta["batch_result_csv_stored"] = True
                meta["batch_result_csv_omitted_reason"] = None
                meta["predictions_in_database"] = True
                js = json.dumps(meta, default=str)
                if len(js) <= INPUT_JSON_MAX_CHARS:
                    c.execute(
                        "UPDATE prediction_logs SET input_json = ? WHERE id = ?",
                        (js, log_id),
                    )
            except json.JSONDecodeError:
                pass
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "message": "Scored CSV saved; open Customer tables again to view."}


def generate_recommendations(
    data: Dict[str, Any], risk_category: str, top_features: List[tuple]
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []

    if risk_category == "LOW RISK":
        recommendations.append(
            {
                "type": "success",
                "title": "Excellent Credit Profile - AUTO APPROVE",
                "message": "Congratulations! Your credit profile demonstrates strong financial responsibility with minimal default risk. Continue these positive behaviors to maintain your excellent standing.",
            }
        )
        for feature, importance in top_features[:4]:
            if feature in data:
                value = data[feature]
                if feature == "SECTOR":
                    recommendations.append(
                        {
                            "type": "maintain",
                            "title": "Industry Sector Stability",
                            "message": "Continue building expertise and stability in your industry to maintain favorable credit terms.",
                        }
                    )
                elif feature == "Age":
                    if value >= 30:
                        recommendations.append(
                            {
                                "type": "maintain",
                                "title": "Mature Credit Profile",
                                "message": f"At {int(value)} years, your age reflects financial maturity. This demographic stability positively impacts your creditworthiness.",
                            }
                        )
                    else:
                        recommendations.append(
                            {
                                "type": "maintain",
                                "title": "Building Credit History",
                                "message": f"At {int(value)} years, continue building your credit history. Your current profile is strong - maintain this trajectory as you mature financially.",
                            }
                        )
                elif feature == "Exposure_Amount":
                    if value <= 300000:
                        recommendations.append(
                            {
                                "type": "maintain",
                                "title": "Conservative Loan Request",
                                "message": "Your loan request demonstrates responsible borrowing aligned with your financial capacity. This conservative approach strengthens your application.",
                            }
                        )
                    else:
                        recommendations.append(
                            {
                                "type": "maintain",
                                "title": "Loan Amount Assessment",
                                "message": "Your loan request has been evaluated against your financial profile. Ensure the amount aligns with your repayment capacity for optimal approval.",
                            }
                        )
                elif feature == "Max_REMAINING_TENOR(AllLoans)":
                    if value <= 24:
                        recommendations.append(
                            {
                                "type": "maintain",
                                "title": "Optimal Loan Tenure",
                                "message": "Your loan tenure strategy shows good debt management. Shorter repayment periods reduce long-term risk and demonstrate strong commitment to lenders.",
                            }
                        )
                    else:
                        recommendations.append(
                            {
                                "type": "tip",
                                "title": "Loan Tenure Optimization",
                                "message": "Consider shorter loan tenures when possible to reduce interest costs and improve future credit applications. This demonstrates stronger financial discipline.",
                            }
                        )
                elif feature == "NetIncome":
                    recommendations.append(
                        {
                            "type": "maintain",
                            "title": "Strong Income Base",
                            "message": f"Your net income of KES {value:,.0f} provides solid financial foundation. Continue diversifying income sources for even greater stability.",
                        }
                    )
                elif feature == "CurrentArrears" and value == 0:
                    recommendations.append(
                        {
                            "type": "maintain",
                            "title": "Perfect Payment Record",
                            "message": "Zero arrears is excellent! Continue making all payments on time to preserve your pristine credit history.",
                        }
                    )
        recommendations.append(
            {
                "type": "tip",
                "title": "Maintain Your Excellence",
                "message": "Keep up your timely payments, maintain low debt-to-income ratio, and continue building savings. Consider applying for higher credit limits as your profile strengthens further.",
            }
        )
    elif risk_category == "MEDIUM RISK":
        recommendations.append(
            {
                "type": "warning",
                "title": "Moderate Risk - REFER / APPROVE WITH CONTROLS",
                "message": "Your profile shows moderate risk. Implement these targeted improvements based on the most influential factors to move to low-risk category and unlock better terms.",
            }
        )
        for feature, importance in top_features[:5]:
            if feature in data:
                value = data[feature]
                if feature == "CurrentArrears" and value > 0:
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "Clear Current Arrears Immediately",
                            "message": f"You have KES {value:,.0f} in arrears. This is the #1 priority - clear all outstanding arrears within 30 days to significantly improve your credit score.",
                        }
                    )
                elif feature == "MaxArrears" and value > 3:
                    recommendations.append(
                        {
                            "type": "improve",
                            "title": "Reduce Arrears History",
                            "message": f"Your maximum arrears of {int(value)} months is concerning. Focus on 6+ months of consistent, on-time payments to rebuild trust with lenders.",
                        }
                    )
                elif feature == "NetIncome" and value < 80000:
                    recommendations.append(
                        {
                            "type": "improve",
                            "title": "Increase Income Streams",
                            "message": f"Current income: KES {value:,.0f}. Target: KES 80,000+. Consider side income, salary negotiation, or skill development to boost earning capacity.",
                        }
                    )
                elif feature == "Exposure_Amount":
                    recommendations.append(
                        {
                            "type": "improve",
                            "title": "Optimize Loan Amount",
                            "message": f"Requested amount: KES {value:,.0f}. Consider reducing by 20-30% to improve approval odds and demonstrate conservative borrowing while you strengthen your profile.",
                        }
                    )
                elif feature == "RUNNING_LOANS_COUNT" and value > 2:
                    recommendations.append(
                        {
                            "type": "improve",
                            "title": "Reduce Active Loans",
                            "message": f"You have {int(value)} active loans. Pay off at least one loan before applying for new credit to reduce debt burden and improve approval chances.",
                        }
                    )
                elif feature == "SavingAcctDepositCount" and value < 15:
                    recommendations.append(
                        {
                            "type": "improve",
                            "title": "Increase Savings Activity",
                            "message": f"Current deposits: {int(value)}. Target: 15+ monthly deposits. Regular savings demonstrate financial discipline and improve creditworthiness.",
                        }
                    )
                elif feature == "CustomerTenure" and value < 3:
                    recommendations.append(
                        {
                            "type": "improve",
                            "title": "Build Banking Relationship",
                            "message": f"Tenure: {value:.1f} years. Target: 3+ years. Maintain active relationship with your bank through regular transactions and savings to build trust.",
                        }
                    )
        recommendations.append(
            {
                "type": "tip",
                "title": "90-Day Action Plan",
                "message": "Focus on: (1) Clear all arrears, (2) Make 3 months of on-time payments, (3) Increase savings deposits to 15+, (4) Reduce loan amount if possible. Reapply after 90 days for better terms.",
            }
        )
    else:
        recommendations.append(
            {
                "type": "danger",
                "title": "High Risk Profile - DECLINE",
                "message": "Your application shows high default risk and requires immediate corrective action. Follow this recovery plan based on the critical factors affecting your score.",
            }
        )
        for feature, importance in top_features[:5]:
            if feature in data:
                value = data[feature]
                if feature == "CurrentArrears" and value > 0:
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "URGENT: Clear All Arrears",
                            "message": f"Outstanding arrears: KES {value:,.0f}. This is critically damaging your credit. Negotiate payment plan with lender and clear within 60 days - this is non-negotiable.",
                        }
                    )
                elif feature == "MaxArrears" and value > 5:
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "URGENT: Rebuild Payment History",
                            "message": f"Maximum arrears of {int(value)} months indicates severe payment issues. You need 12+ months of perfect payment history before reapplying. Set up auto-payments immediately.",
                        }
                    )
                elif feature == "NetIncome" and value < 50000:
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "URGENT: Increase Income",
                            "message": f"Income of KES {value:,.0f} is insufficient for requested loan. Delay application until income reaches KES 80,000+ through additional work or business opportunities.",
                        }
                    )
                elif feature == "RUNNING_LOANS_COUNT" and value > 3:
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "URGENT: Reduce Debt Burden",
                            "message": f"{int(value)} active loans is excessive. Pay off at least 2 loans completely before considering new credit. Focus on debt consolidation if possible.",
                        }
                    )
                elif feature == "NonPerforming" and value > 0:
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "URGENT: Resolve Non-Performing Loans",
                            "message": f"You have {int(value)} non-performing loan(s). This severely damages creditworthiness. Resolve immediately through restructuring or settlement.",
                        }
                    )
                elif feature == "Exposure_Amount":
                    recommendations.append(
                        {
                            "type": "critical",
                            "title": "Loan Amount Too High",
                            "message": f"Requested KES {value:,.0f} exceeds your capacity. Reduce by 50%+ or delay application for 6-12 months while improving financial position.",
                        }
                    )
        recommendations.append(
            {
                "type": "tip",
                "title": "6-Month Recovery Plan",
                "message": "Do NOT apply for new credit now. Instead: (1) Clear all arrears within 60 days, (2) Reduce active loans to maximum 2, (3) Build 6 months of perfect payment history, (4) Increase income to 80K+, (5) Save 10% of income monthly. Reapply after 6 months.",
            }
        )
    return recommendations


if __name__ == "__main__":
    import uvicorn

    # 0.0.0.0 is for binding only — browsers cannot open http://0.0.0.0:8000 (use 127.0.0.1 or localhost).
    _host = os.environ.get("UVICORN_HOST", "127.0.0.1")
    # Default 8765: on some Windows setups 8080 is in use or in an excluded TCP range (WinError 10013).
    _port = int(os.environ.get("UVICORN_PORT", "8765"))
    print(f"Open in your browser: http://127.0.0.1:{_port}/  (listening on {_host}:{_port})")
    uvicorn.run("app:app", host=_host, port=_port, reload=True)
