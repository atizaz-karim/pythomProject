import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("patient_health.db")
SQL_PATH = Path("sql/create_tables.sql")
CSV_PATH = Path("sample_data/patient_data.csv")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    with open(SQL_PATH, "r") as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()
    conn.close()
    print("Database initialized at", DB_PATH)

def load_sample_csv():
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    conn = sqlite3.connect(DB_PATH)

    # Insert patients if not exists
    for name in df["patient_name"].unique():
        cur = conn.execute("SELECT patient_id FROM patients WHERE patient_name=?", (name,))
        if cur.fetchone() is None:
            conn.execute("INSERT INTO patients (patient_name) VALUES (?)", (name,))
    conn.commit()

    # Map patient names to their IDs
    patients = {r[0]: r[1] for r in conn.execute("SELECT patient_name, patient_id FROM patients")}

    # Insert metrics
    rows = []
    for _, row in df.iterrows():
        pid = patients[row["patient_name"]]
        rows.append((
            pid,
            row["timestamp"].isoformat(),
            row.get("heart_rate"),
            row.get("systolic_bp"),
            row.get("diastolic_bp"),
            row.get("temperature"),
            None
        ))

    conn.executemany(
        "INSERT INTO metrics (patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp, temperature, ecg_blob) VALUES (?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()
    print("Sample CSV loaded into database.")

if __name__ == '__main__':
    init_db()
    load_sample_csv()