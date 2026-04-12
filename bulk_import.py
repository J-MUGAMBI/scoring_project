"""
Bulk import CSV data into the local SQLite database (users.db).

Two modes:

  1) raw  — Stores any spreadsheet as its own table (no ML model).
            Use this for "All Graduated spreadsheet..." and similar files whose
            columns are NOT the credit model features.

  2) credit — Runs the credit-risk model and creates prediction_logs +
              batch_prediction_rows (same as Batch Scoring in the app).
              The CSV must contain every column listed in feature_list.json.

Examples (run from this folder, venv activated):

  python bulk_import.py raw --csv "All Graduated spreadsheet - Sheet1 (1) - All Graduated spreadsheet - Sheet1 (1).csv"

  python bulk_import.py credit --csv my_credit_features.csv --user-id 1

Find your user id:  python bulk_import.py list-users

Clear bad data + load the bundled credit sample in one step:

  python bulk_import.py reset-load-sample --user-id 1

Or manually:

  python bulk_import.py clear-uploads --yes
  python bulk_import.py credit --csv X_deploy_sample.csv --user-id 1
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_db_resolved = os.path.abspath(
    os.environ.get("SQLITE_DB_PATH", os.path.join(BASE_DIR, "users.db"))
)
_db_dir = os.path.dirname(_db_resolved)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)
DB_PATH = _db_resolved

DEFAULT_GRADUATED_CSV = (
    "All Graduated spreadsheet - Sheet1 (1) - All Graduated spreadsheet - Sheet1 (1).csv"
)

SAMPLE_CREDIT_CSV = "X_deploy_sample.csv"

# Tables we never DROP (schema + users). All other user tables are removed by clear-uploads.
_CORE_KEEP = frozenset(
    {"users", "sqlite_sequence", "prediction_logs", "batch_prediction_rows", "bulk_import_meta"}
)


def _sanitize_table_name(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name).strip("_").lower()
    if not s or s[0].isdigit():
        s = "imported_" + s
    return s[:60] or "imported_data"


def cmd_clear_uploads(*, assume_yes: bool) -> None:
    """Remove all prediction logs, per-row batch data, bulk-import metadata, and extra import tables."""
    if not assume_yes:
        print(
            "This deletes ALL prediction_logs, batch_prediction_rows, bulk_import_meta, "
            "and any raw-import tables (graduated spreadsheet, etc.). Users are kept.\n"
            "Re-run with --yes to confirm.",
            file=sys.stderr,
        )
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        c = conn.cursor()
        c.execute("DELETE FROM batch_prediction_rows")
        c.execute("DELETE FROM prediction_logs")
        try:
            c.execute("DELETE FROM bulk_import_meta")
        except sqlite3.OperationalError:
            pass
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (name,) in c.fetchall():
            if name not in _CORE_KEEP:
                c.execute(f'DROP TABLE IF EXISTS "{name}"')
        for tbl in ("prediction_logs", "batch_prediction_rows"):
            try:
                c.execute("DELETE FROM sqlite_sequence WHERE name = ?", (tbl,))
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()
    print(f"Cleared uploads in {DB_PATH} (user accounts unchanged).")


def cmd_list_users() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT id, username, full_name, email FROM users ORDER BY id")
        rows = c.fetchall()
    finally:
        conn.close()
    if not rows:
        print("No users in users.db — register via the web app first.")
        return
    print("id | username | full_name | email")
    for r in rows:
        print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]}")


def cmd_raw_import(csv_path: str, table_name: str | None) -> None:
    path = csv_path if os.path.isabs(csv_path) else os.path.join(BASE_DIR, csv_path)
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    tbl = table_name or _sanitize_table_name(os.path.splitext(os.path.basename(path))[0])
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        # Replace table on each full reload so re-running the script refreshes data
        df.to_sql(tbl, conn, if_exists="replace", index=False)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS bulk_import_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                source_file TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL
            )"""
        )
        c.execute(
            "INSERT INTO bulk_import_meta (table_name, source_file, row_count, imported_at) VALUES (?,?,?,?)",
            (
                tbl,
                os.path.basename(path),
                len(df),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    print(f"Imported {len(df)} rows into table '{tbl}' in {DB_PATH}")


def cmd_credit_batch(csv_path: str, user_id: int) -> None:
    """Same scoring + DB writes as /batch_predict (requires model feature columns)."""
    os.chdir(BASE_DIR)
    sys.path.insert(0, BASE_DIR)

    import numpy as np

    try:
        import app as app_mod
    except ImportError as e:
        print(
            "Failed to load the ML app (model + sklearn). Use the project's virtualenv:\n"
            "  .\\venv\\Scripts\\activate   (Windows)\n"
            "  python bulk_import.py credit --csv X_deploy_sample.csv --user-id 1\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(3)

    path = csv_path if os.path.isabs(csv_path) else os.path.join(BASE_DIR, csv_path)
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    missing = [f for f in app_mod.features if f not in df.columns]
    if missing:
        print(
            "This CSV does not contain the credit model's input columns.\n"
            f"Missing ({len(missing)}): {missing[:15]}"
            + (" ..." if len(missing) > 15 else "")
            + "\n\n"
            "The graduated student spreadsheet uses different columns (grades, attendance, …).\n"
            "Use:  python bulk_import.py raw --csv \"...graduated....csv\"\n"
            "to store it in SQLite without predictions.\n\n"
            "For prediction_logs from the model, supply a CSV that matches feature_list.json.",
            file=sys.stderr,
        )
        sys.exit(2)

    input_data = df[app_mod.features]
    probabilities = app_mod.model.predict_proba(input_data)[:, 1]
    predictions = (probabilities >= app_mod.BEST_THRESHOLD).astype(int)
    decisions = [app_mod.decision_from_pd(float(p)) for p in probabilities]
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

    log_id = app_mod.insert_batch_prediction_log_transaction(
        user_id,
        threshold=app_mod.BEST_THRESHOLD,
        batch_total=len(df),
        batch_high=high_risk,
        batch_med=medium_risk,
        batch_low=low_risk,
        batch_pred_default=n_pred_default,
        batch_pred_no_default=n_pred_no_default,
        input_payload={
            "filename": os.path.basename(path),
            "rows": len(df),
            "threshold": app_mod.BEST_THRESHOLD,
            "predicted_default_risk_rows": n_pred_default,
            "predicted_no_default_rows": n_pred_no_default,
            "risk_tiers": {
                "HIGH RISK": high_risk,
                "MEDIUM RISK": medium_risk,
                "LOW RISK": low_risk,
            },
            "predictions_in_database": True,
            "import_source": "bulk_import.py credit",
        },
        results_df=results_df,
        predictions=predictions,
        risk_categories=risk_categories,
    )
    if log_id is None:
        print("Failed to write prediction_logs / batch_prediction_rows.", file=sys.stderr)
        sys.exit(1)
    print(
        f"OK — batch log id={log_id}, rows={len(df)}. "
        f"Open Prediction Logs in the app (user id {user_id}) to view tables."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Bulk import CSV into SQLite / credit batch")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_clear = sub.add_parser(
        "clear-uploads",
        help="Delete all predictions, batch rows, and raw-import tables (keeps users)",
    )
    p_clear.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually delete (safety)",
    )

    p_users = sub.add_parser("list-users", help="List user ids for --user-id")

    p_raw = sub.add_parser("raw", help="Import any CSV into its own table (no model)")
    p_raw.add_argument(
        "--csv",
        default=DEFAULT_GRADUATED_CSV,
        help=f"Path to CSV (default: {DEFAULT_GRADUATED_CSV})",
    )
    p_raw.add_argument(
        "--table",
        default=None,
        help="SQLite table name (default: derived from filename)",
    )

    p_cr = sub.add_parser(
        "credit",
        help="Run credit model + write prediction_logs (CSV must match feature_list.json)",
    )
    p_cr.add_argument("--csv", required=True, help="Path to credit-features CSV")
    p_cr.add_argument("--user-id", type=int, required=True, help="Owner user id (see list-users)")

    p_sync = sub.add_parser(
        "reset-load-sample",
        help=f"clear-uploads --yes then import {SAMPLE_CREDIT_CSV} (credit model)",
    )
    p_sync.add_argument("--user-id", type=int, required=True, help="Owner user id (see list-users)")

    args = p.parse_args()

    if args.cmd == "clear-uploads":
        cmd_clear_uploads(assume_yes=args.yes)
    elif args.cmd == "list-users":
        cmd_list_users()
    elif args.cmd == "raw":
        cmd_raw_import(args.csv, args.table)
    elif args.cmd == "credit":
        cmd_credit_batch(args.csv, args.user_id)
    elif args.cmd == "reset-load-sample":
        sample = os.path.join(BASE_DIR, SAMPLE_CREDIT_CSV)
        if not os.path.isfile(sample):
            print(f"Missing {sample}", file=sys.stderr)
            sys.exit(1)
        cmd_clear_uploads(assume_yes=True)
        cmd_credit_batch(sample, args.user_id)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
