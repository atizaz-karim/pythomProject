import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.fft import rfft, rfftfreq

# Try to import your project modules (fall back to safe local funcs)
try:
    from signal_processing import moving_average, apply_bandpass
except Exception:
    # fallback moving average
    def moving_average(x, w=3):
        if len(x) < 1: return x
        return np.convolve(x, np.ones(w)/w, mode='same')
    def apply_bandpass(*args, **kwargs):
        raise RuntimeError("apply_bandpass not available")

try:
    from image_processing import load_image, enhance_contrast, detect_edges, blur_image, to_grayscale
except Exception:
    # fallback simple image ops using PIL / cv2 if available
    from PIL import Image, ImageFilter, ImageOps
    def load_image(path):
        return Image.open(path).convert("RGB")
    def enhance_contrast(img):
        return ImageOps.autocontrast(img)
    def detect_edges(img):
        return img.convert("L").filter(ImageFilter.FIND_EDGES)
    def blur_image(img, radius=2):
        return img.filter(ImageFilter.GaussianBlur(radius))
    def to_grayscale(img):
        return img.convert("L")

# ---------- Database helpers (sqlite) ----------
DB_PATH = "patienthealth.db"

def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def ensure_tables():
    conn = get_db_conn()
    cur = conn.cursor()
    # simple metrics table (if you already have one, this will add nothing)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient TEXT,
        timestamp TEXT,
        metric_name TEXT,
        metric_value REAL
    )
    """)
    # images table for raw/processed images
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient TEXT,
        description TEXT,
        mime TEXT,
        image BLOB,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def fetch_metrics_db(patient=None, start=None, end=None):
    """
    Returns a DataFrame with columns: timestamp, patient, <metric_name columns pivoted>
    This helper expects metrics stored in long format (metric_name, metric_value).
    If your existing DB schema is different, adapt this function.
    """
    conn = get_db_conn()
    q = "SELECT patient, timestamp, metric_name, metric_value FROM metrics"
    params = []
    clauses = []
    if patient and patient != "All":
        clauses.append("patient = ?")
        params.append(patient)
    if start:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        clauses.append("timestamp <= ?")
        params.append(end)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(q, conn, params=params, parse_dates=["timestamp"])
    conn.close()
    if df.empty:
        return pd.DataFrame()
    # pivot to wide format: index timestamp, columns metric_name
    df_wide = df.pivot_table(index="timestamp", columns="metric_name", values="metric_value", aggfunc="mean").reset_index()
    # keep patient if present (first unique)
    if "patient" in df.columns:
        df_wide["patient"] = df["patient"].iloc[0]
    # Ensure timestamp column is datetime
    df_wide['timestamp'] = pd.to_datetime(df_wide['timestamp'])
    return df_wide

def save_image_to_db(patient, description, pil_image, mime="image/png"):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b = buf.getvalue()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO images (patient, description, mime, image) VALUES (?, ?, ?, ?)", (patient, description, mime, b))
    conn.commit()
    conn.close()

def fetch_images_for_patient(patient=None):
    conn = get_db_conn()
    q = "SELECT id, patient, description, mime, image, created_at FROM images"
    params = []
    if patient and patient != "All":
        q += " WHERE patient = ?"
        params.append(patient)
    q += " ORDER BY created_at DESC"
    df = pd.read_sql_query(q, conn, params=params, parse_dates=["created_at"])
    conn.close()
    return df

# ---------- UI: Data Visualization Page ----------
def data_visualization_page():
    ensure_tables()
    st.header("Data Visualization")

    # Controls: patient, date range
    col1, col2 = st.columns([2, 1])
    with col1:
        patient = st.selectbox("Select patient", options=get_patient_options(), index=0)
    with col2:
        # simple date range inputs (optional)
        start = st.date_input("Start date", value=None)
        end = st.date_input("End date", value=None)

    df = fetch_metrics_db(patient=patient if patient else None,
                          start=start.isoformat() if start else None,
                          end=end.isoformat() if end else None)

    if df.empty:
        st.info("No metrics found for that patient / date range. You can upload data to the DB or load from Patient Data Management section.")
    else:
        st.subheader("Preview of loaded data")
        st.dataframe(df.head(50))

    st.markdown("---")
    st.subheader("1) Time-series: raw and filtered")

    if df.empty:
        st.info("Load metric data first.")
    else:
        metric_cols = [c for c in df.columns if c not in ("timestamp","patient")]
        metric_choice = st.selectbox("Select metric", metric_cols)
        ts = pd.to_datetime(df["timestamp"])
        y = df[metric_choice].astype(float)

        # layout for raw + filtered
        col_r, col_f = st.columns(2)
        with col_r:
            st.markdown("**Raw signal**")
            st.line_chart(pd.DataFrame({metric_choice: y.values}, index=ts))
        with col_f:
            st.markdown("**Filtered / Smoothed**")
            w = st.slider("Moving average window", 1, 51, 3, step=2)
            smooth = moving_average(y.values, w=w)
            st.line_chart(pd.DataFrame({f"{metric_choice}_smooth": smooth}, index=ts))

            # optional bandpass
            try:
                if st.checkbox("Apply bandpass filter (requires dense sampling)"):
                    low = st.number_input("Low cut (Hz)", value=0.5, step=0.1)
                    high = st.number_input("High cut (Hz)", value=40.0, step=0.1)
                    fs = st.number_input("Sampling rate (Hz)", value=1.0, step=0.1)
                    filtered = apply_bandpass(y.values.astype(float), lowcut=low, highcut=high, fs=fs)
                    st.line_chart(pd.DataFrame({f"{metric_choice}_bandpass": filtered}, index=ts))
            except Exception as e:
                st.warning("Bandpass filter failed: " + str(e))

    st.markdown("---")
    st.subheader("2) Scatter plot: relationship between metrics")

    if df.empty:
        st.info("Load metric data first.")
    else:
        cols = metric_cols
        x = st.selectbox("X metric", cols, index=0)
        y_metric = st.selectbox("Y metric", cols, index=1 if len(cols)>1 else 0)
        fig = px.scatter(df, x=x, y=y_metric, hover_data=["timestamp"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3) Heatmap: correlation matrix")

    if df.empty:
        st.info("Load metric data first.")
    else:
        corr = df[[c for c in metric_cols]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("4) Spectrum Analysis (FFT)")

    if df.empty:
        st.info("Load metric data first.")
    else:
        metric_fft_choice = st.selectbox("Metric for FFT", metric_cols, index=0)
        signal = df[metric_fft_choice].dropna().astype(float).values
        if len(signal) < 2:
            st.warning("Signal too short for FFT.")
        else:
            # assume uniform sampling; frequency axis approximate using rfftfreq
            N = len(signal)
            fs = st.number_input("Approx sampling frequency (Hz)", value=1.0, step=0.1)
            yf = np.abs(rfft(signal))
            xf = rfftfreq(N, d=1.0/fs)
            fig2, ax2 = plt.subplots()
            ax2.plot(xf, yf)
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Amplitude")
            ax2.set_title(f"Spectrum of {metric_fft_choice}")
            st.pyplot(fig2)

    st.markdown("---")
    st.subheader("5) Image Display & Processing")

    img_patient = st.selectbox("Select patient for images", options=get_patient_options())
    uploaded = st.file_uploader("Upload medical image (store to DB)", type=["png","jpg","jpeg"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(uploaded.getvalue())
        tmp.flush()
        pil_img = load_image(tmp.name)
        desc = st.text_input("Image description", value="uploaded image")
        if st.button("Save image to DB"):
            save_image_to_db(img_patient, desc, pil_img)
            st.success("Image saved.")

    # show images
    imgs_df = fetch_images_for_patient(img_patient)
    if imgs_df.empty:
        st.info("No images found for this patient.")
    else:
        for _, row in imgs_df.iterrows():
            st.markdown(f"**{row['description']}** — {row['created_at']}")
            # convert blob to image
            im = None
            try:
                im = load_pil_from_blob(row["image"])
            except Exception:
                im = None
            if im:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(im, caption="Original", use_column_width=True)
                with col2:
                    # basic processing controls
                    op = st.selectbox(f"Process (id={row['id']})", ["Grayscale","Blur","Edges"], key=f"proc_{row['id']}")
                    if op == "Grayscale":
                        proc = to_grayscale(im)
                    elif op == "Blur":
                        proc = blur_image(im, radius=3)
                    else:
                        proc = detect_edges(im)
                    st.image(proc, caption=f"Processed ({op})", use_column_width=True)
                    if st.button(f"Save processed image to DB (id={row['id']})", key=f"saveproc_{row['id']}"):
                        save_image_to_db(img_patient, f"{row['description']} ({op})", proc)
                        st.success("Processed image saved.")

# ---------- helper utils ----------
def get_patient_options():
    # If you have patients table elsewhere, adapt. For now read distinct patient names from metrics/images.
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT patient FROM metrics")
    rows = cur.fetchall()
    conn.close()
    names = ["All"]
    for r in rows:
        if r[0]:
            names.append(r[0])
    # also ensure at least one default
    if len(names) == 1:
        names += ["Atizaz","Karim","Rashid"]
    return names

def load_pil_from_blob(blob):
    """Return a PIL Image from sqlite blob (bytes)"""
    from PIL import Image
    import io
    if blob is None:
        return None
    b = blob
    if isinstance(b, memoryview):
        b = b.tobytes()
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return img
