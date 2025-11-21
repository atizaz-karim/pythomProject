import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import sqlite3
from pathlib import Path

DB_PATH = Path("patient_health.db")


# ----------------- Time-Domain Signal Processing -----------------
def moving_average(x, w=5):
    """
    Simple moving average smoothing
    """
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='same')


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Butterworth bandpass filter design
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass(signal, lowcut=0.5, highcut=40.0, fs=250.0):
    """
    Apply bandpass filter to a 1D signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = filtfilt(b, a, signal)
    return y


# ----------------- Frequency-Domain / FFT Analysis -----------------
def compute_fft(signal, sample_spacing=1.0):
    """
    Compute FFT of a 1D signal
    Returns frequencies and amplitudes
    """
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, sample_spacing)[:N//2]
    amplitudes = 2.0 / N * np.abs(yf[0:N//2])
    return xf, amplitudes


def store_fft_in_db(patient_id, metric, xf, amplitudes):
    """
    Store FFT frequency components in the database table 'frequency_analysis'
    """
    # Ensure table exists
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS frequency_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        metric TEXT,
        frequency REAL,
        amplitude REAL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(create_table_sql)
    for f, a in zip(xf, amplitudes):
        conn.execute(
            "INSERT INTO frequency_analysis (patient_id, metric, frequency, amplitude) VALUES (?,?,?,?)",
            (patient_id, metric, f, a)
        )
    conn.commit()
    conn.close()


def fetch_fft_from_db(patient_id, metric):
    """
    Fetch stored FFT data for a patient and metric
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT frequency, amplitude 
    FROM frequency_analysis 
    WHERE patient_id=? AND metric=?
    ORDER BY frequency ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ----------------- Script / Demo -----------------
if __name__ == '__main__':
    # Time-domain demo
    t = np.linspace(0, 1.0, 250)
    sig = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(len(t))

    print("Original mean:", np.mean(sig))

    smooth = moving_average(sig, w=5)
    filtered = apply_bandpass(sig, lowcut=1, highcut=40, fs=250)

    print("Time-domain smoothing & bandpass done.")

    # FFT demo
    xf, yf = compute_fft(sig)
    print("FFT computed. First 10 frequencies & amplitudes:")
    for f, a in zip(xf[:10], yf[:10]):
        print(f"{f:.2f} Hz : {a:.3f}")
