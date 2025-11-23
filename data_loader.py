import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("patient_data.db")
CSV_PATH = Path("heart_disease.csv")  # <- your CSV file

# ------------------------------
# Fetch metrics from CSV (instead of DB)
# ------------------------------
def fetch_metrics_csv():
    """
    Load patient metrics from CSV file.
    Returns a pandas DataFrame.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found.")
    
    df = pd.read_csv(CSV_PATH)
    
    # Ensure columns are compatible with app
    required_cols = ["patient_id", "date", "heart_rate", "oxygen", "resp_rate", "bp_sys", "bp_dia"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing columns with 0
    
    return df

# ------------------------------
# Clean metrics
# ------------------------------
def clean_metrics(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["heart_rate", "oxygen", "resp_rate", "bp_sys", "bp_dia"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols, how="all")
    df = df.sort_values("date")
    return df
