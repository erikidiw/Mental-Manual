# FILE: app.py (VERSI KOREKSI UNTUK TypeError)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- NAMA KOLOM KONSTAN (HARUS SAMA PERSIS DENGAN DATASET) ---
COL_SUICIDAL = 'Have you ever had suicidal thoughts?'
COL_SLEEP = 'Sleep Duration'
COL_FINANCIAL = 'Financial Stress'
COL_FAMILY = 'Family History of Mental Illness'
COL_STUDY_SAT = 'Study Satisfaction' 

# --- 1. LOAD ASSETS ---
try:
    best_gb = joblib.load('gb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Memuat dictionary Label Encoders
    le_encoders = joblib.load('label_encoder.pkl') 
    te = joblib.load('target_encoder.pkl')
    
    # TITIK KRITIS: MENGAKSES OBJEK DARI DICTIONARY
    DEGREES = list(le_encoders['Degree'].classes_)
    DIETARY_HABITS = list(le_encoders['Dietary Habits'].classes_)
    GENDER_OPTIONS = list(le_encoders['Gender'].classes_) 

except FileNotFoundError:
    st.error("Error: Berkas model (.pkl) tidak ditemukan. Mohon pastikan semua 4 berkas PKL sudah diunggah.")
    st.stop()
except KeyError as e:
    # Menangani jika kunci Degree, Gender, atau Dietary Habits hilang
    st.error(f"Error: Kunci {e} hilang di file 'label_encoder.pkl'. Mohon buat ulang file PKL.")
    st.stop()
except AttributeError:
    # Menangani jika objek yang dimuat BUKAN LabelEncoder (e.g., jika isinya None atau tipe lain)
    st.error("Error: Objek yang disimpan di 'label_encoder.pkl' mungkin rusak atau bukan LabelEncoder.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file PKL: {e}")
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
st.title("Mental Health Predictor for Students 🧠")
st.markdown("Aplikasi prediksi potensi depresi menggunakan model **Optimized Gradient Boosting Classifier**.")
st.write("---")

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
    study_satisfaction = st.slider("Study Satisfaction (Skala 0-5)", 0.0, 5.0, 3.0)
    sleep_duration_input = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()))
    dietary_habits_input = st.selectbox("Dietary Habits", DIETARY_HABITS)


with col_c:
    st.header("3. Riwayat & Lainnya")
    degree_input = st.selectbox("Degree", DEGREES)
    profession_input = st.text_input("Profession", PROFESSION_DEFAULT)
    city_input = st.text_input("City", CITY_DEFAULT)
    financial_stress_input = st.selectbox("Financial Stress (1=Rendah, 5=Tinggi)", FINANCIAL_OPTIONS)
    suicidal_thoughts_input = st.selectbox("Pernah punya pikiran bunuh diri?", ["No", "Yes"])
    family_history_input = st.selectbox("Riwayat Keluarga Gangguan Mental", ["No", "Yes"])


st.write("---")

# --- 4. PREDICTION LOGIC ---

if st.button("Prediksi Potensi Depresi"):
    # 1. Kumpulkan data input dengan NAMA KOLOM YANG AKURAT
    data = {
        'Gender': [gender_input], 
        'Age': [age], 
        'City': [city_input], 
        'Profession': [profession_input],
        'Academic Pressure': [academic_pressure], 
        'CGPA': [cgpa], 
        COL_STUDY_SAT: [study_satisfaction],
        COL_SLEEP: [sleep_duration_input], 
        'Dietary Habits': [dietary_habits_input], 
        'Degree': [degree_input],
        COL_SUICIDAL: [suicidal_thoughts_input], 
        'Work/Study Hours': [work_study_hours], 
        COL_FINANCIAL: [financial_stress_input], 
        COL_FAMILY: [family_history_input]
    }
    input_df = pd.DataFrame(data)
    
    # 2. Lakukan Encoding sesuai urutan

    # a. Binary/Ordinal Mapping
    input_df[COL_SLEEP] = input_df[COL_SLEEP].map(SLEEP_MAP).fillna(0.0)
    input_df[COL_FINANCIAL] = input_df[COL_FINANCIAL].map(FINANCIAL_MAP).fillna(0.0)
    input_df[COL_SUICIDAL] = input_df[COL_SUICIDAL].map(SUICIDAL_MAP).fillna(0.0)
    input_df[COL_FAMILY] = input_df[COL_FAMILY].map(FAMILY_MAP).fillna(0.0)
    
    # b. Label Encoding
    label_cols_transform = ['Gender', 'Dietary Habits', 'Degree']
    for col in label_cols_transform:
        le = le_encoders[col] 
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError:
            st.error(f"Error: Kategori input '{input_df[col].iloc[0]}' pada kolom '{col}' tidak dikenal.")
            st.stop()
            
    # c. Target Encoding
    input_df[['City', 'Profession']] = te.transform(input_df[['City', 'Profession']])
    
    # d. Scaling
    feature_order = scaler.feature_names_in_
    try:
        input_scaled = scaler.transform(input_df[feature_order])
    except KeyError as e:
        st.error(f"FATAL KEY ERROR: Kolom {e} tidak ditemukan di DataFrame input.")
        st.warning(f"Kolom yang diharapkan Scaler: {list(feature_order)}")
        st.warning(f"Kolom yang ada di Input: {list(input_df.columns)}")
        st.stop()

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
