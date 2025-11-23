# import sqlite3

# conn = sqlite3.connect("patient_data.db")
# cur = conn.cursor()

# cur.execute("""
# CREATE TABLE IF NOT EXISTS metrics (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     patient_id TEXT,
#     date TEXT,
#     heart_rate REAL,
#     oxygen REAL,
#     resp_rate REAL,
#     bp_sys REAL,
#     bp_dia REAL
# )
# """)

# conn.commit()
# conn.close()

# print("Database initialized. Table 'metrics' created.")
