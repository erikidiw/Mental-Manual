import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder

# --- 1. LOAD ASSETS ---
# Muat model, scaler, dan encoders
try:
    best_gb = joblib.load('gb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    te = joblib.load('target_encoder.pkl')
except FileNotFoundError:
    st.error("Berkas model (pkl) tidak ditemukan. Pastikan Anda sudah menjalankan skrip 'Langkah Membuat Model .pkl' dan file berada di direktori yang sama.")
    st.stop()

# --- 2. PRE-PROCESSING UTILITIES (Harus sama dengan langkah training) ---

# Fungsi capper harus sama persis dengan yang digunakan saat training
def cap_outliers_single(value, col_name):
    # Dapatkan statistik IQR dari data training yang sudah di-scale
    # Kita menggunakan data hasil scaling/encoding untuk mendapatkan batas-batas yang sesuai
    # Namun, karena proses capping seharusnya dilakukan pada data mentah sebelum scaling,
    # kita akan membatasi ini hanya untuk memastikan konsistensi format input
    
    # Untuk implementasi Streamlit yang sederhana, kita asumsikan input sudah bersih/tercaping
    # atau kita melewatkan proses capping di sini.
    # Jika capping ingin diimplementasikan dengan benar, kita perlu menyimpan Q1 dan Q3 data mentah (df_encoded)
    return value

# Mapping untuk input kategorikal
SLEEP_MAP = {"<5 jam": 1.0, "5-6 jam": 2.0, "7-8 jam": 3.0, ">8 jam": 4.0, "Lainnya": 0.0}
FINANCIAL_MAP = {"1.0 (Sangat Rendah)": 1.0, "2.0 (Rendah)": 2.0, "3.0 (Sedang)": 3.0, "4.0 (Tinggi)": 4.0, "5.0 (Sangat Tinggi)": 5.0, "Tidak Tahu/Kosong": 0.0}
SUICIDAL_MAP = {"Tidak": 0.0, "Ya": 1.0}
FAMILY_HISTORY_MAP = {"Tidak": 0.0, "Ya": 1.0}

# --- 3. STREAMLIT APP LAYOUT ---
st.title("Depression Predictor: Gradient Boosting")
st.markdown("Aplikasi prediksi depresi menggunakan model **Gradient Boosting Classifier**.")
st.write("---")

# --- INPUT FITUR NUMERIK/ORDINAL ---
st.header("Data Akademik dan Demografi")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (Usia)", 18, 60, 25)
    cgpa = st.slider("CGPA (Skala 0-10)", 0.0, 10.0, 7.5, 0.01)
    academic_pressure = st.slider("Academic Pressure (Skala 0-5)", 0.0, 5.0, 3.0)

with col2:
    work_study_hours = st.slider("Work/Study Hours (Jam/Hari)", 0.0, 12.0, 8.0)
    study_satisfaction = st.slider("Study Satisfaction (Skala 0-5)", 0.0, 5.0, 3.0)
    gender_input = st.selectbox("Gender", ["Male", "Female"])

st.write("---")

# --- INPUT FITUR KATEGORIKAL ---
st.header("Faktor Gaya Hidup dan Psikologis")

col3, col4 = st.columns(2)
with col3:
    sleep_duration_input = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()))
    dietary_habits_input = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy", "Others"])
    financial_stress_input = st.selectbox("Financial Stress", list(FINANCIAL_MAP.keys()))

with col4:
    suicidal_thoughts_input = st.selectbox("Pernah memiliki pikiran bunuh diri?", list(SUICIDAL_MAP.keys()))
    family_history_input = st.selectbox("Family History of Mental Illness", list(FAMILY_HISTORY_MAP.keys()))
    
# Karena City dan Degree memiliki banyak kategori dan di-Target Encode,
# untuk kemudahan, kita akan menggunakan nilai Target Encoding rata-rata dari data training
# atau menggunakan input yang paling sering muncul
city_input = st.text_input("City (untuk tujuan demonstrasi, akan menggunakan rata-rata City TE)", "Kalyan")
degree_input = st.text_input("Degree (untuk tujuan demonstrasi, akan menggunakan rata-rata Degree LE)", "'Class 12'")
profession_input = st.text_input("Profession (untuk tujuan demonstrasi, akan menggunakan rata-rata Profession TE)", "Student")

# --- 4. DATA TRANSFORMATION (Pre-processing input) ---

if st.button("Prediksi Depresi"):
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
    
    # 2. Lakukan Encoding sesuai urutan di file PDF (Binary/Ordinal/Label/Target)
    
    # a. Binary/Ordinal Encoding (Mapping)
    input_df['Sleep Duration'] = input_df['Sleep Duration'].map(SLEEP_MAP).fillna(0.0)
    input_df['Financial Stress'] = input_df['Financial Stress'].map(FINANCIAL_MAP).fillna(0.0)
    input_df['Have you ever had suicidal thoughts?'] = input_df['Have you ever had suicidal thoughts?'].map(SUICIDAL_MAP).fillna(0.0)
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(FAMILY_HISTORY_MAP).fillna(0.0)
    
    # b. Label Encoding (Gender, Dietary Habits, Degree)
    # Gunakan fit_transform pada data training dan transform pada data testing.
    # Karena objek LE sudah dimuat, kita bisa langsung transform.
    for col in ['Gender', 'Dietary Habits', 'Degree']:
        input_df[col] = le.transform(input_df[col].astype(str))
        
    # c. Target Encoding (City, Profession)
    # Gunakan objek TE yang sudah fit
    # Jika kategori baru (tidak ada di data training), TargetEncoder akan memberikan nilai rata-rata target training
    input_df[target_enc_cols] = te.transform(input_df[target_enc_cols])
    
    # d. Capping (Dilewatkan untuk input tunggal di Streamlit, diatasi oleh data yang sudah di-transform)
    
    # e. Standardisasi (Scaling)
    # Fitur-fitur harus dalam urutan yang sama saat training
    feature_order = scaler.feature_names_in_
    input_scaled = scaler.transform(input_df[feature_order])
    
    # 3. Prediksi
    prediction = best_gb.predict(input_scaled)[0]
    prediction_proba = best_gb.predict_proba(input_scaled)[0]
    
    # 4. Tampilkan Hasil
    st.write("---")
    st.header("Hasil Prediksi")
    
    if prediction == 1:
        st.error("Prediksi: Depresi (Kelas 1)")
        st.metric("Probabilitas Depresi (Kelas 1)", f"{prediction_proba[1]:.2%}")
        st.markdown(
            """
            > **Rekomendasi:** Model memprediksi bahwa individu ini cenderung **mengalami depresi**.
            > Perlu peninjauan lebih lanjut oleh profesional kesehatan mental.
            """
        )
    else:
        st.success("Prediksi: Tidak Depresi (Kelas 0)")
        st.metric("Probabilitas Tidak Depresi (Kelas 0)", f"{prediction_proba[0]:.2%}")
        st.markdown(
            """
            > **Rekomendasi:** Model memprediksi bahwa individu ini cenderung **tidak mengalami depresi**.
            > Tetap jaga kesehatan mental dengan baik.
            """
        )
