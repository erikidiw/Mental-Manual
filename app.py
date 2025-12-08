# FILE: app.py (VERSI FINAL TERKOREKSI UNTUK SEMUA FITUR DAN KEYERROR)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD ASSETS & GET FEATURE NAMES ---
try:
    best_gb = joblib.load('gb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_encoders = joblib.load('label_encoder.pkl') 
    te = joblib.load('target_encoder.pkl')
    
    FEATURE_NAMES = list(scaler.feature_names_in_)
    
    # Ambil nama kolom yang kompleks dari daftar yang dipelajari Scaler
    def get_feature_name(partial_name):
        matches = [col for col in FEATURE_NAMES if partial_name in col]
        return matches[0] if matches else partial_name

    # Menggunakan nama kolom yang terverifikasi dari Scaler untuk INPUT/MAPPING:
    COL_SUICIDAL = get_feature_name('suicidal thoughts')
    COL_SLEEP = get_feature_name('Sleep Duration')
    COL_FINANCIAL = get_feature_name('Financial Stress')
    COL_FAMILY = get_feature_name('Family History')
    COL_STUDY_SAT = get_feature_name('Study Satisfaction')
    COL_ACADEMIC_PRESSURE = get_feature_name('Academic Pressure')
    COL_WORK_STUDY_HOURS = get_feature_name('Work/Study Hours')
    
    # Ambil daftar kelas untuk SelectBox dari dictionary Label Encoder
    DEGREES = list(le_encoders['Degree'].classes_)
    DIETARY_HABITS = list(le_encoders['Dietary Habits'].classes_)
    GENDER_OPTIONS = list(le_encoders['Gender'].classes_) 

except Exception as e:
    st.error(f"Terjadi kesalahan fatal saat memuat file PKL atau mendapatkan Feature Names: {e}")
    st.warning("Pastikan Anda sudah membuat ulang file PKL dengan skrip create_pkl_files.py terbaru dan mengunggahnya.")
    st.stop()


# --- 2. MAPPINGS & DEFAULTS ---
SLEEP_MAP = {"Less than 5 hours": 1.0, "5-6 hours": 2.0, "7-8 hours": 3.0, "More than 8 hours": 4.0, "Others": 0.0}
FINANCIAL_MAP = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "?": 0.0} 
SUICIDAL_MAP = {"No": 0.0, "Yes": 1.0}
FAMILY_MAP = {"No": 0.0, "Yes": 1.0}

FINANCIAL_OPTIONS = ["1", "2", "3", "4", "5", "?"] 
CITY_DEFAULT = "Kalyan"
PROFESSION_DEFAULT = "Student"


# --- 3. STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide")
st.title("Mental Health Predictor: UJI BOBOT SEMUA FITUR 🧠")
st.markdown("Fitur bobot tinggi (Suicide, Academic Pressure) akan menyebabkan perubahan besar, sementara fitur bobot kecil (Gender, CGPA) akan menyebabkan perubahan kecil.")
st.write("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.header("1. Demografi & Akademik")
    gender_input = st.selectbox("Gender", GENDER_OPTIONS)
    age = st.slider("Age (Usia)", 18, 60, 25)
    cgpa = st.slider("CGPA (Skala 0-10)", 0.0, 10.0, 7.5, 0.01)
    academic_pressure = st.slider("Academic Pressure", 0.0, 5.0, 3.0) # Bobot Tinggi

with col_b:
    st.header("2. Gaya Hidup & Stres")
    work_study_hours = st.slider("Work/Study Hours", 0.0, 12.0, 8.0) # Bobot Menengah
    study_satisfaction = st.slider("Study Satisfaction", 0.0, 5.0, 3.0)
    sleep_duration_input = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()))
    dietary_habits_input = st.selectbox("Dietary Habits", DIETARY_HABITS)


with col_c:
    st.header("3. Riwayat & Lainnya")
    degree_input = st.selectbox("Degree", DEGREES)
    profession_input = st.text_input("Profession", PROFESSION_DEFAULT)
    city_input = st.text_input("City", CITY_DEFAULT)
    financial_stress_input = st.selectbox("Financial Stress", FINANCIAL_OPTIONS) # Bobot Menengah
    suicidal_thoughts_input = st.selectbox("Pernah punya pikiran bunuh diri?", ["No", "Yes"]) # Bobot Tertinggi
    family_history_input = st.selectbox("Riwayat Keluarga Gangguan Mental", ["No", "Yes"])


st.write("---")

# --- 4. PREDICTION LOGIC ---

if st.button("PREDIKSI & UJI PENGARUH"):
    # 1. Kumpulkan data input dengan KUNCI yang diambil dari Scaler
    data = {
        'Gender': [gender_input], 
        'Age': [age], 
        'City': [city_input], 
        'Profession': [profession_input],
        COL_ACADEMIC_PRESSURE: [academic_pressure], 
        'CGPA': [cgpa], 
        COL_STUDY_SAT: [study_satisfaction],
        COL_SLEEP: [sleep_duration_input], 
        'Dietary Habits': [dietary_habits_input], 
        'Degree': [degree_input],
        COL_SUICIDAL: [suicidal_thoughts_input], 
        COL_WORK_STUDY_HOURS: [work_study_hours], 
        COL_FINANCIAL: [financial_stress_input], 
        COL_FAMILY: [family_history_input]
    }
    input_df = pd.DataFrame(data)
    
    # 2. Lakukan Encoding (Mapping/Ordinal)
    input_df[COL_SLEEP] = input_df[COL_SLEEP].map(SLEEP_MAP).fillna(0.0)
    input_df[COL_FINANCIAL] = input_df[COL_FINANCIAL].map(FINANCIAL_MAP).fillna(0.0)
    input_df[COL_SUICIDAL] = input_df[COL_SUICIDAL].map(SUICIDAL_MAP).fillna(0.0)
    input_df[COL_FAMILY] = input_df[COL_FAMILY].map(FAMILY_MAP).fillna(0.0)
    
    # b. Label Encoding
    label_cols_transform = ['Gender', 'Dietary Habits', 'Degree']
    for col in label_cols_transform:
        le = le_encoders[col] 
        input_df[col] = le.transform(input_df[col].astype(str))
            
    # c. Target Encoding
    input_df[['City', 'Profession']] = te.transform(input_df[['City', 'Profession']])
    
    # d. Scaling (Menggunakan urutan kolom yang pasti benar dari Scaler)
    input_scaled = scaler.transform(input_df[FEATURE_NAMES])

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
