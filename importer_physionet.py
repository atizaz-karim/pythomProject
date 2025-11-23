# importer_physionet.py
import wfdb
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.signal import find_peaks
from datetime import datetime, timedelta

# ---------------- Database path ----------------
DB_PATH = Path("patient_health.db")
CSV_PATH = Path("physionet_patient_data.csv")

# ---------------- Download BIDMC Dataset ----------------
# This dataset has multiple recordings: BIDMC PPG and Respiration
# Each recording has signals sampled at 125Hz
# We'll download a few sample records
record_ids = ["bpm001", "bpm002", "bpm003"]  # small sample for demonstration

all_data = []

for patient_id, rec_id in enumerate(record_ids, start=1):
    print(f"Processing patient {patient_id} - {rec_id}")
    # download record
    record = wfdb.rdrecord(rec_id, pn_dir='bidmc')
    signal = record.p_signal[:, 0]  # use first channel (PPG)
    fs = record.fs

    # ---------------- Extract Heart Rate ----------------
    # Simple peak detection
    peaks, _ = find_peaks(signal, distance=fs*0.5)  # min 0.5 sec apart
    timestamps = [record.sig_name[0]] * len(peaks)
    start_time = datetime.now()  # arbitrary start time
    times = [start_time + timedelta(seconds=i*(1/fs)*fs) for i in range(len(peaks))]  # approx 1Hz heart rate

    hr = np.random.randint(60, 100, size=len(peaks))  # simulated heart rate

    # ---------------- Simulate BP and Temp ----------------
    systolic_bp = np.random.randint(110, 130, size=len(peaks))
    diastolic_bp = np.random.randint(70, 85, size=len(peaks))
    temperature = np.random.normal(36.5, 0.3, size=len(peaks))

    # ---------------- Compile patient data ----------------
    df_patient = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": times,
        "heart_rate": hr,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "temperature": temperature
    })
    all_data.append(df_patient)

# ---------------- Combine all patients ----------------
df_all = pd.concat(all_data, ignore_index=True)

# ---------------- Save CSV ----------------
df_all.to_csv(CSV_PATH, index=False)
print(f"CSV saved: {CSV_PATH}")

# ---------------- Insert into SQLite DB ----------------
conn = sqlite3.connect(DB_PATH)

# Create tables if not exist
conn.execute("""
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY,
    patient_name TEXT
);
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    timestamp TEXT,
    heart_rate REAL,
    systolic_bp REAL,
    diastolic_bp REAL,
    temperature REAL,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);
""")

# Insert patients
for pid in df_all["patient_id"].unique():
    conn.execute("INSERT OR IGNORE INTO patients (patient_id, patient_name) VALUES (?, ?)", 
                 (pid, f"Patient {pid}"))

# Insert metrics
for _, row in df_all.iterrows():
    conn.execute("""
    INSERT INTO metrics (patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp, temperature)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (row.patient_id, row.timestamp, row.heart_rate, row.systolic_bp, row.diastolic_bp, row.temperature))

conn.commit()
conn.close()

print("All data imported into SQLite database successfully!")
