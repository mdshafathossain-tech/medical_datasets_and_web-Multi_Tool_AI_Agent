"""
db_setup.py
------------
Handles creation and population of SQLite databases for the Multi-Tool
Medical AI Agent project.

Reads raw CSVs from `data/`, performs light cleaning, and loads each
dataset into its own SQLite database under `databases/`.

Run directly to (re)build all databases:
    python db_setup.py
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_DIR = Path("data")
DB_DIR = Path("databases")

# Maps: dataset name -> (csv filename, db filename, table name)
DATASET_CONFIG = {
    "heart": {
        "csv": DATA_DIR / "heart.csv",
        "db": DB_DIR / "heart.db",
        "table": "heart_data",
    },
    "cancer": {
        "csv": DATA_DIR / "cancer.csv",
        "db": DB_DIR / "cancer.db",
        "table": "cancer_data",
    },
    "diabetes": {
        "csv": DATA_DIR / "diabetes.csv",
        "db": DB_DIR / "diabetes.db",
        "table": "diabetes_data",
    },
}


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, strip whitespace,
    replace spaces/hyphens with underscores.

    This keeps generated SQL predictable for the text-to-SQL agent later.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)  # drop any remaining odd chars
    )
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light, generic cleaning suitable for these medical CSVs:
      - Standardize column names
      - Replace common missing-value placeholders ('?', '', 'NA') with NaN
      - Drop rows that are entirely empty
      - Attempt to convert numeric-looking columns to numeric dtype
    """
    df = clean_column_names(df)

    # Common placeholder values for missing data in UCI-style medical datasets
    df = df.replace(["?", "", "NA", "N/A", "na", "n/a"], pd.NA)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Try converting columns to numeric where possible (leaves text columns alone)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only replace if conversion didn't wipe out all values
        # (i.e., the column really was numeric-like)
        if converted.notna().sum() > 0 and converted.notna().sum() >= df[col].notna().sum() * 0.5:
            df[col] = converted

    return df


# --------------------------------------------------------------------------
# Core DB build logic
# --------------------------------------------------------------------------

def build_database(dataset_name: str, force_rebuild: bool = False) -> None:
    """
    Build a single SQLite database from its source CSV.

    Args:
        dataset_name: One of 'heart', 'cancer', 'diabetes'.
        force_rebuild: If True, rebuilds the DB even if it already exists.
    """
    config = DATASET_CONFIG[dataset_name]
    csv_path: Path = config["csv"]
    db_path: Path = config["db"]
    table_name: str = config["table"]

    # Ensure the databases/ directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not force_rebuild:
        print(f"[skip] '{db_path}' already exists. Use force_rebuild=True to recreate it.")
        return

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected CSV not found: '{csv_path}'. "
            f"Please place '{csv_path.name}' inside the '{DATA_DIR}/' folder."
        )

    print(f"[build] Reading '{csv_path}' ...")
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    print(f"[build] Writing {len(df)} rows to '{db_path}' (table: '{table_name}') ...")
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    print(f"[done] '{db_path}' created successfully.\n")


def build_all_databases(force_rebuild: bool = False) -> None:
    """Build all 3 databases (heart, cancer, diabetes)."""
    for dataset_name in DATASET_CONFIG:
        build_database(dataset_name, force_rebuild=force_rebuild)


# --------------------------------------------------------------------------
# Schema inspection helper (used later by tools.py for text-to-SQL prompts)
# --------------------------------------------------------------------------

def get_table_schema(db_path: Path, table_name: str) -> str:
    """
    Return a human-readable schema string for a given table, e.g.:
        "age (REAL), sex (INTEGER), cp (INTEGER), chol (REAL), target (INTEGER)"

    This will be injected into the LLM prompt so it knows the exact
    column names/types when generating SQL queries.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
    finally:
        conn.close()

    schema_str = ", ".join(f"{col[1]} ({col[2]})" for col in columns)
    return schema_str


def print_all_schemas() -> None:
    """Utility: print the schema of every dataset's table (for debugging)."""
    for dataset_name, config in DATASET_CONFIG.items():
        db_path = config["db"]
        table_name = config["table"]
        if not db_path.exists():
            print(f"[warn] '{db_path}' does not exist yet. Run build_all_databases() first.")
            continue
        schema = get_table_schema(db_path, table_name)
        print(f"{dataset_name} ({table_name}):\n  {schema}\n")


# --------------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building all medical databases from CSVs in 'data/' ...\n")
    build_all_databases(force_rebuild=False)
    print("Schemas:\n")
    print_all_schemas()