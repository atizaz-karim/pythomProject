import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from data_loader import fetch_metrics_csv, clean_metrics

# ---------------------------------------------------------
# Custom Sidebar Styles
st.markdown("""
<style>

/* Sidebar container size */
[data-testid="stSidebar"] {
    width: 355px !important;        
    padding: 10px;
}

/* Sidebar title (Navigation) */
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    font-size: 22px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 12px !important;
}
label[data-baseweb="radio"] {
    display: flex !important;
    align-items: center !important;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Page Title
st.title("Healthcare Data & Medical Image Processing Tool")

# ---------------------------------------------------------
# Sidebar Navigation
section = st.sidebar.radio(
    "Navigation",
    [
        "Patient Data Management",
        "Health Data Analysis",
        "Spectrum Analysis",
        "Image Processing",
        "Data Visualization"
    ]
)

# ---------------------------------------------------------
# Global shared dataframe (session state)
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None


# =========================================================
# 📌 SECTION 1 — PATIENT DATA MANAGEMENT
# =========================================================
if section == "Patient Data Management":
    st.header("Patient Data Management")
    st.write("Load raw and cleaned patient metrics from the CSV file.")

    # Load CSV data
    df = fetch_metrics_csv()

    patient_options = ['All'] + df['patient_id'].unique().tolist()
    patient = st.selectbox('Select patient', options=patient_options)

    # Convert CSV 'date' column to datetime.date
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    start = st.date_input('Start date', value=df['date'].min())
    end = st.date_input('End date', value=df['date'].max())

    if st.button("Load Data"):
        df_clean = clean_metrics(df)

        # Apply patient filter
        if patient != "All":
            df_clean = df_clean[df_clean['patient_id'] == patient]

        # Apply date filter
        df_clean = df_clean[(df_clean['date'] >= pd.to_datetime(start)) & 
                            (df_clean['date'] <= pd.to_datetime(end))]

        st.session_state.df_clean = df_clean

        st.write("### Raw Data (first 10 rows)")
        st.dataframe(df.head(10))

        st.write("### Cleaned & Filtered Data")
        st.dataframe(df_clean.head(10))


# =========================================================
# 📌 SECTION 2 — HEALTH DATA ANALYSIS
# =========================================================
elif section == "Health Data Analysis":
    st.header("Health Data Analysis")
    st.write("Visualize heart-rate data and apply signal filters.")

    df_clean = st.session_state.df_clean

    if df_clean is None or df_clean.empty:
        st.info("➡ Load patient data first from **Patient Data Management**.")
    else:
        # Ensure 'date' is datetime64 for plotting
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['heart_rate', 'date'])
        df_clean = df_clean.sort_values('date')

        ts = df_clean['date']
        hr = df_clean['heart_rate'].astype(float)

        if hr.empty:
            st.warning("No heart rate data to display.")
        else:
            # RAW Heart Rate
            st.subheader("Raw Heart Rate")
            st.line_chart(pd.DataFrame({"HR": hr}, index=ts))

            # Moving Average Smooth
            smooth = hr.rolling(window=5, min_periods=1).mean()
            st.subheader("Smoothed (Moving Average)")
            st.line_chart(pd.DataFrame({"HR Smooth": smooth}, index=ts))



# =========================================================
# 📌 SECTION 3 — SPECTRUM ANALYSIS
# =========================================================
elif section == "Spectrum Analysis":
    st.header("Spectrum Analysis (FFT)")
    st.write("Analyze frequency components of heart-rate signals.")

    df_clean = st.session_state.df_clean

    if df_clean is None or df_clean.empty:
        st.info("➡ Load patient data first.")
    else:
        hr = df_clean["heart_rate"].astype(float).values

        # FFT Spectrum
        st.subheader("Frequency Spectrum")
        n = len(hr)
        freqs = np.fft.fftfreq(n, d=1.0)
        spectrum = np.abs(np.fft.fft(hr))

        fig, ax = plt.subplots()
        ax.plot(freqs[:n // 2], spectrum[:n // 2])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

# =========================================================
# 📌 SECTION 4 — IMAGE PROCESSING
# =========================================================
elif section == "Image Processing":
    st.header("Medical Image Processing")
    st.write("Upload a medical image and apply processing steps.")

    img_file = st.file_uploader("Upload medical image", type=["png", "jpg", "jpeg"])

    if img_file:
        import tempfile
        import cv2
        import numpy as np

        # Save uploaded file temporarily
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_file.getvalue())
        tmp.flush()

        # Load image with OpenCV headless
        img = cv2.imread(tmp.name)
        if img is None:
            st.error("Failed to load image.")
        else:
            # Convert BGR → RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display original
            st.subheader("Original Image")
            st.image(img_rgb, use_column_width=True)

            # Enhanced contrast (CLAHE)
            st.subheader("Enhanced Contrast")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_gray)
            st.image(img_enhanced, use_column_width=True, clamp=True)

            # Edge detection (Canny)
            st.subheader("Edge Detection")
            edges = cv2.Canny(img_enhanced, 50, 150)
            st.image(edges, use_column_width=True, clamp=True)




# =========================================================
# 📌 SECTION 5 — DATA VISUALIZATION
# =========================================================
elif section == "Data Visualization":
    st.header("Data Visualization")
    st.write("Explore relationships between health metrics.")

    df_clean = st.session_state.df_clean

    if df_clean is None or df_clean.empty:
        st.info("➡ Load patient data first.")
    else:
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

        # Time-series plot
        st.subheader("Time-Series Plot")
        metric = st.selectbox("Select metric", numeric_cols)
        ts = pd.to_datetime(df_clean["date"])
        st.line_chart(pd.DataFrame({metric: df_clean[metric]}, index=ts))

        # Scatter plot
        st.subheader("Scatter Plot")
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
        fig, ax = plt.subplots()
        ax.scatter(df_clean[x_col], df_clean[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
