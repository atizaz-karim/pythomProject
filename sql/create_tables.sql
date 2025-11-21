-- SQLite schema for patient health data
CREATE TABLE IF NOT EXISTS patients (
patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
patient_name TEXT,
dob TEXT
);


CREATE TABLE IF NOT EXISTS metrics (
id INTEGER PRIMARY KEY AUTOINCREMENT,
patient_id INTEGER,
timestamp TEXT,
heart_rate REAL,
systolic_bp REAL,
diastolic_bp REAL,
temperature REAL,
ecg_blob TEXT,
FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);


CREATE TABLE IF NOT EXISTS images (
id INTEGER PRIMARY KEY AUTOINCREMENT,
patient_id INTEGER,
path TEXT,
modality TEXT,
metadata TEXT,
FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);