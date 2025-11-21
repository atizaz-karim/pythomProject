import streamlit as st
import pandas as pd
from data_loader import fetch_metrics, clean_metrics
from signal_processing import moving_average, apply_bandpass
from image_processing import load_image, enhance_contrast, detect_edges

st.title('Patient Health Data Processing')

patient = st.selectbox('Select patient', options=['All','Atizaz','Karim','Rashid'])
start = st.date_input('Start date')
end = st.date_input('End date')

if st.button('Load data'):
    if patient == 'All':
        df = fetch_metrics()
    else:
        df = fetch_metrics(patient_name=patient, start=start.isoformat(), end=end.isoformat())
    st.write('Raw data (first 10 rows):')
    st.dataframe(df.head(10))

    df_clean = clean_metrics(df)
    st.write('Cleaned data:')
    st.dataframe(df_clean.head(10))

    if not df_clean.empty:
        hr = df_clean['heart_rate'].values
        ts = pd.to_datetime(df_clean['timestamp'])
        st.line_chart(pd.DataFrame({'heart_rate': hr}, index=ts))

        # smoothing
        smooth = moving_average(hr, w=3)
        st.line_chart(pd.DataFrame({'hr_smooth': smooth}, index=ts))

        # bandpass demo if ECG-like data present (here we reuse hr as dummy)
        try:
            filtered = apply_bandpass(hr.astype(float), lowcut=0.5, highcut=40, fs=1.0)
            st.line_chart(pd.DataFrame({'hr_filtered': filtered}, index=ts))
        except Exception as e:
            st.warning('Bandpass failed (needs denser signal).')

st.header('Image Processing')
img_file = st.file_uploader('Upload X-ray / image', type=['png','jpg','jpeg'])
if img_file is not None:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    tmp.write(img_file.getvalue())
    tmp.flush()
    img = load_image(tmp.name)
    st.image(img, caption='Original', use_column_width=True)
    enhanced = enhance_contrast(img)
    st.image(enhanced, caption='Enhanced', use_column_width=True)
    edges = detect_edges(enhanced)
    st.image(edges, caption='Edges', use_column_width=True)