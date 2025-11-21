import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("patient_health.db")


def fetch_metrics(patient_name=None, start=None, end=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT m.*, p.patient_name FROM metrics m JOIN patients p ON m.patient_id = p.patient_id"
    filters = []
    params = []
    if patient_name:
        filters.append("p.patient_name = ?")
        params.append(patient_name)
    if start:
        filters.append("timestamp >= ?")
        params.append(start)
    if end:
        filters.append("timestamp <= ?")
        params.append(end)
    if filters:
        query += " WHERE " + " AND ".join(filters)
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
    conn.close()
    return df


def clean_metrics(df):
    # simple cleaning: drop rows with missing mandatory metrics
    df = df.dropna(subset=["heart_rate", "systolic_bp", "diastolic_bp"])
    # remove unrealistic values using thresholds
    df = df[(df["heart_rate"] > 30) & (df["heart_rate"] < 200)]
    df = df[(df["systolic_bp"] > 50) & (df["systolic_bp"] < 250)]
    return df


# ----------------- Correlation Analysis -----------------
def compute_correlations(df, method='pearson'):
    """
    Compute correlation matrix for selected metrics.
    method: 'pearson' or 'spearman'
    """
    metrics = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature']
    corr = df[metrics].corr(method=method)
    return corr


def store_correlations_in_db(corr, method='pearson'):
    """
    Store pairwise correlations into the database table 'correlations'.
    """
    # Ensure table exists
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS correlations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_x TEXT,
        metric_y TEXT,
        method TEXT,
        coefficient REAL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(create_table_sql)
    for x in corr.columns:
        for y in corr.columns:
            if x != y:
                conn.execute(
                    "INSERT INTO correlations (metric_x, metric_y, method, coefficient) VALUES (?,?,?,?)",
                    (x, y, method, corr.loc[x, y])
                )
    conn.commit()
    conn.close()


def fetch_correlations(method='pearson'):
    """
    Fetch stored correlation data from the database.
    """
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT metric_x, metric_y, coefficient FROM correlations WHERE method=?"
    df = pd.read_sql_query(query, conn, params=(method,))
    conn.close()
    return df


# ----------------- Script execution -----------------
if __name__ == '__main__':
    df = fetch_metrics()
    print("Raw metrics:\n", df.head())

    df_clean = clean_metrics(df)
    print("\nAfter cleaning:\n", df_clean.head())

    # Compute correlations
    corr_matrix = compute_correlations(df_clean, method='pearson')
    print("\nPearson correlation matrix:\n", corr_matrix)

    # Store in DB
    store_correlations_in_db(corr_matrix, method='pearson')
    print("\nCorrelations stored in the database.")

    # Fetch stored correlations
    fetched_corr = fetch_correlations(method='pearson')
    print("\nFetched correlations from DB:\n", fetched_corr.head())
