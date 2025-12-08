# FILE: app.py (VERSI KOREKSI AKHIR)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder

# --- 1. LOAD ASSETS ---
try:
    # Memuat Model dan Scaler
    best_gb = joblib.load('gb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # MEMUAT LABEL ENCODER DARI FILE 'label_encoder.pkl' (BENTUK DICTIONARY)
    le_encoders = joblib.load('label_encoder.pkl') 
    # Memuat Target Encoder
    te = joblib.load('target_encoder.pkl')
    
except FileNotFoundError:
    st.error("Error: Berkas model (.pkl) tidak ditemukan. Mohon pastikan semua 4 berkas PKL sudah diunggah ke repositori.")
    st.stop()

# --- 2. MAPPINGS & DEFAULTS ---

SLEEP_MAP = {"Less than 5 hours": 1.0, "5-6 hours": 2.0, "7-8 hours": 3.0, "More than 8 hours": 4.0, "Others": 0.0}
# PENTING: Nilai input string untuk Financial Stress harus disesuaikan dengan yang di-fit, yaitu "1", "2", dst.
FINANCIAL_MAP = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "?": 0.0} 
SUICIDAL_MAP = {"No": 0.0, "Yes": 1.0}
FAMILY_HISTORY_MAP = {"No": 0.0, "Yes": 1.0}

# Ambil daftar kelas yang valid untuk SelectBox
DEGREES = list(le_encoders['Degree'].classes_)
DIETARY_HABITS = list(le_encoders['Dietary Habits'].classes_)
GENDER_OPTIONS = list(le_encoders['Gender'].classes_) 
FINANCIAL_OPTIONS = ["1", "2", "3", "4", "5", "?"] 
CITY_DEFAULT = "Kalyan"
PROFESSION_DEFAULT = "Student"


# --- 3. STREAMLIT APP LAYOUT (Disajikan lebih ringkas) ---
st.set_page_config(layout="wide")
st.title("Mental Health Predictor for Students 🧠")
st.markdown("Aplikasi prediksi potensi depresi menggunakan model **Optimized Gradient Boosting Classifier**.")
st.write("---")

# Input Group
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.header("1. Demografi & Akademik")
    gender_input = st.selectbox("Gender", GENDER_OPTIONS)
    age = st.slider("Age (Usia)", 18, 60, 25)
    cgpa = st.slider("CGPA (Skala 0-10)", 0.0, 10.0, 7.5, 0.01)
    academic_pressure = st.slider("Academic Pressure (Skala 0-5)", 0.0, 5.0, 3.0)

with col_b:
    st.header("2. Gaya Hidup & Stres")
    work_study_hours = st.slider("Work/Study Hours (Jam/Hari)", 0.0, 12.0, 8.0)
    sleep_duration_input = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()))
    dietary_habits_input = st.selectbox("Dietary Habits", DIETARY_HABITS)
    financial_stress_input = st.selectbox("Financial Stress (1=Rendah, 5=Tinggi)", FINANCIAL_OPTIONS)

with col_c:
    st.header("3. Status & Riwayat")
    degree_input = st.selectbox("Degree", DEGREES)
    profession_input = st.text_input("Profession", PROFESSION_DEFAULT)
    city_input = st.text_input("City", CITY_DEFAULT)
    suicidal_thoughts_input = st.selectbox("Pernah punya pikiran bunuh diri?", ["No", "Yes"])
    family_history_input = st.selectbox("Riwayat Keluarga Gangguan Mental", ["No", "Yes"])

st.write("---")

# --- 4. PREDICTION LOGIC ---

if st.button("Prediksi Potensi Depresi"):
    data = {
        'Gender': [gender_input], 'Age': [age], 'City': [city_input], 'Profession': [profession_input],
        'Academic Pressure': [academic_pressure], 'CGPA': [cgpa], 'Study Satisfaction': [st.slider("Study Satisfaction (Skala 0-5)", 0.0, 5.0, 3.0)],
        'Sleep Duration': [sleep_duration_input], 'Dietary Habits': [dietary_habits_input], 'Degree': [degree_input],
        'Have you ever had suicidal thoughts?': [suicidal_thoughts_input], 'Work/Study Hours': [work_study_hours], 
        'Financial Stress': [financial_stress_input], 'Family History of Mental Illness': [family_history_input]
    }
    input_df = pd.DataFrame(data)
    
    # a. Binary/Ordinal Encoding (Mapping)
    input_df['Sleep Duration'] = input_df['Sleep Duration'].map(SLEEP_MAP).fillna(0.0)
    input_df['Financial Stress'] = input_df['Financial Stress'].map(FINANCIAL_MAP).fillna(0.0)
    input_df['Have you ever had suicidal thoughts?'] = input_df['Have you ever had suicidal thoughts?'].map(SUICIDAL_MAP).fillna(0.0)
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(FAMILY_HISTORY_MAP).fillna(0.0)
    
    # b. Label Encoding (Menggunakan multiple encoders)
    label_cols_transform = ['Gender', 'Dietary Habits', 'Degree']
    for col in label_cols_transform:
        le = le_encoders[col] 
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError as e:
            # Karena input adalah selectbox, ini seharusnya tidak terjadi, 
            # tapi tetap ada untuk memastikan robustness
            st.error(f"Error internal pada kolom {col}. Coba pilih opsi yang berbeda.")
            st.stop()
            
    # c. Target Encoding
    input_df[['City', 'Profession']] = te.transform(input_df[['City', 'Profession']])
    
    # d. Scaling
    feature_order = scaler.feature_names_in_
    input_scaled = scaler.transform(input_df[feature_order])
    
    # 3. Prediksi
    prediction = best_gb.predict(input_scaled)[0]
    prediction_proba = best_gb.predict_proba(input_scaled)[0]
    
    # 4. Tampilkan Hasil
    st.write("---")
    st.header("Hasil Prediksi")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 1:
            st.error("Status: Depresi (Kelas 1)")
        else:
            st.success("Status: Tidak Depresi (Kelas 0)")
            
    with col_res2:
        st.metric("Probabilitas Depresi (Kelas 1)", f"{prediction_proba[1]:.2%}")
        st.metric("Probabilitas Tidak Depresi (Kelas 0)", f"{prediction_proba[0]:.2%}")
