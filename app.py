# FILE: app.py
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
    # Memuat multiple Label Encoders (disimpan dalam dictionary)
    le_encoders = joblib.load('label_encoders.pkl')
    # Memuat Target Encoder
    te = joblib.load('target_encoder.pkl')
    
except FileNotFoundError:
    st.error("Error: Berkas model (.pkl) tidak ditemukan. Pastikan 4 berkas PKL diunggah.")
    st.stop()

# --- 2. MAPPINGS & DEFAULTS ---

# Mapping input: Harus sesuai dengan yang digunakan saat training
SLEEP_MAP = {"Less than 5 hours": 1.0, "5-6 hours": 2.0, "7-8 hours": 3.0, "More than 8 hours": 4.0, "Others": 0.0}
# Financial Stress input perlu disesuaikan karena nilai training adalah string "1.0", "2.0", dst
FINANCIAL_MAP = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "?": 0.0}
SUICIDAL_MAP = {"No": 0.0, "Yes": 1.0}
FAMILY_HISTORY_MAP = {"No": 0.0, "Yes": 1.0}

# Ambil daftar kelas yang valid untuk SelectBox
DEGREES = list(le_encoders['Degree'].classes_)
DIETARY_HABITS = list(le_encoders['Dietary Habits'].classes_)
GENDER_OPTIONS = ["Male", "Female"]
FINANCIAL_OPTIONS = ["1", "2", "3", "4", "5", "?"] # Menampilkan input sesuai string yang digunakan saat training

# --- 3. STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide")
st.title("Mental Health Predictor for Students 🧠")
st.markdown("Aplikasi prediksi potensi depresi menggunakan model **Optimized Gradient Boosting Classifier**.")
st.write("---")

# --- INPUT FITUR NUMERIK/ORDINAL ---
st.header("1. Data Akademik dan Demografi")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age (Usia)", 18, 60, 25)
    academic_pressure = st.slider("Academic Pressure (Skala 0-5)", 0.0, 5.0, 3.0)

with col2:
    cgpa = st.slider("CGPA (Skala 0-10)", 0.0, 10.0, 7.5, 0.01)
    study_satisfaction = st.slider("Study Satisfaction (Skala 0-5)", 0.0, 5.0, 3.0)

with col3:
    work_study_hours = st.slider("Work/Study Hours (Jam/Hari)", 0.0, 12.0, 8.0)
    gender_input = st.selectbox("Gender", GENDER_OPTIONS)

st.write("---")

# --- INPUT FITUR KATEGORIKAL ---
st.header("2. Faktor Gaya Hidup dan Psikologis")

col4, col5, col6 = st.columns(3)
with col4:
    sleep_duration_input = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()))
    dietary_habits_input = st.selectbox("Dietary Habits", DIETARY_HABITS)
    financial_stress_input = st.selectbox("Financial Stress (1=Rendah, 5=Tinggi)", FINANCIAL_OPTIONS)

with col5:
    degree_input = st.selectbox("Degree", DEGREES)
    profession_input = st.text_input("Profession (e.g., Student, Lawyer, Architect)", "Student")

with col6:
    suicidal_thoughts_input = st.selectbox("Pernah memiliki pikiran bunuh diri?", list(SUICIDAL_MAP.keys()))
    family_history_input = st.selectbox("Family History of Mental Illness", list(FAMILY_HISTORY_MAP.keys()))
    city_input = st.text_input("City (e.g., Kalyan, Delhi)", "Kalyan")

# --- 4. DATA TRANSFORMATION ---

if st.button("Prediksi Potensi Depresi"):
    # 1. Kumpulkan data input
    data = {
        'Gender': [gender_input],
        'Age': [age],
        'City': [city_input],
        'Profession': [profession_input],
        'Academic Pressure': [academic_pressure],
        'CGPA': [cgpa],
        'Study Satisfaction': [study_satisfaction],
        'Sleep Duration': [sleep_duration_input],
        'Dietary Habits': [dietary_habits_input],
        'Degree': [degree_input],
        'Have you ever had suicidal thoughts?': [suicidal_thoughts_input],
        'Work/Study Hours': [work_study_hours],
        'Financial Stress': [financial_stress_input],
        'Family History of Mental Illness': [family_history_input]
    }
    input_df = pd.DataFrame(data)
    
    # 2. Lakukan Encoding sesuai urutan

    # a. Binary/Ordinal Encoding (Mapping)
    input_df['Sleep Duration'] = input_df['Sleep Duration'].map(SLEEP_MAP).fillna(0.0)
    input_df['Financial Stress'] = input_df['Financial Stress'].map(FINANCIAL_MAP).fillna(0.0)
    input_df['Have you ever had suicidal thoughts?'] = input_df['Have you ever had suicidal thoughts?'].map(SUICIDAL_MAP).fillna(0.0)
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(FAMILY_HISTORY_MAP).fillna(0.0)
    
    # b. Label Encoding (Menggunakan multiple encoders)
    label_cols_transform = ['Gender', 'Dietary Habits', 'Degree']
    for col in label_cols_transform:
        le = le_encoders[col] # Ambil encoder spesifik
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError as e:
            # Penanganan untuk input yang belum terlihat (Unseen Labels)
            st.error(f"Error pada kolom {col}: Kategori '{input_df[col].iloc[0]}' tidak dikenal oleh model.")
            st.warning("Silakan pilih nilai dari daftar opsi yang tersedia atau pastikan input ejaan sudah benar.")
            st.stop()
            
    # c. Target Encoding
    # TargetEncoder akan secara otomatis menangani kategori baru dengan menggantinya 
    # dengan nilai rata-rata target training.
    input_df[target_enc_cols] = te.transform(input_df[target_enc_cols])
    
    # d. Standardisasi (Scaling)
    feature_order = scaler.feature_names_in_
    input_scaled = scaler.transform(input_df[feature_order])
    
    # 3. Prediksi
    prediction = best_gb.predict(input_scaled)[0]
    prediction_proba = best_gb.predict_proba(input_scaled)[0]
    
    # 4. Tampilkan Hasil
    st.write("---")
    st.header("Hasil Prediksi Model")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 1:
            st.error("Status: Depresi (Kelas 1)")
            st.markdown(
                """
                > **Interpretasi:** Model memprediksi individu ini cenderung **mengalami depresi**.
                >
                > **Rekomendasi:** Berdasarkan data, **Academic Pressure**, **Suicidal Thoughts**, dan **Financial Stress** memiliki korelasi tertinggi dengan Depresi. Pertimbangkan untuk mencari dukungan profesional.
                """
            )
        else:
            st.success("Status: Tidak Depresi (Kelas 0)")
            st.markdown(
                """
                > **Interpretasi:** Model memprediksi individu ini cenderung **tidak mengalami depresi**.
                >
                > **Rekomendasi:** Tetap jaga keseimbangan **Work/Study Hours** dan **Dietary Habits** untuk mempertahankan kondisi mental yang sehat.
                """
            )
            
    with col_res2:
        st.metric("Probabilitas Depresi (Kelas 1)", f"{prediction_proba[1]:.2%}")
        st.metric("Probabilitas Tidak Depresi (Kelas 0)", f"{prediction_proba[0]:.2%}")
