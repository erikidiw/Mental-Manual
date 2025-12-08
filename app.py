# FILE: app.py (VERSI FINAL TERKOREKSI DENGAN SELECTBOX)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# 🔧 Custom Preprocessing Classes
# ==========================
class CustomOrdinalMapper:
    def __init__(self, mappings):
        if isinstance(mappings, list):
            self.mappings = {col: map_dict for col, map_dict in mappings}
        else:
            self.mappings = mappings
            
        self.cols = list(self.mappings.keys())
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping).fillna(0).astype(float)
        return X_copy[self.cols]


# --- 1. LOAD ASSETS & GET FEATURE NAMES ---
try:
    # Memuat semua artefak yang diperlukan
    best_gb = joblib.load('gb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_encoders = joblib.load('label_encoder.pkl') 
    te = joblib.load('target_encoder.pkl')
    
    FEATURE_NAMES = list(scaler.feature_names_in_)
    
    def get_feature_name(partial_name):
        matches = [col for col in FEATURE_NAMES if partial_name in col]
        return matches[0] if matches else partial_name

    COL_SUICIDAL = get_feature_name('suicidal thoughts')
    COL_SLEEP = get_feature_name('Sleep Duration')
    COL_FINANCIAL = get_feature_name('Financial Stress')
    COL_FAMILY = get_feature_name('Family History')
    COL_STUDY_SAT = get_feature_name('Study Satisfaction')
    COL_ACADEMIC_PRESSURE = get_feature_name('Academic Pressure')
    COL_WORK_STUDY_HOURS = get_feature_name('Work/Study Hours')
    
    # Ambil daftar kelas untuk SelectBox
    DEGREES = list(le_encoders['Degree'].classes_)
    DIETARY_HABITS = list(le_encoders['Dietary Habits'].classes_)
    GENDER_OPTIONS = list(le_encoders['Gender'].classes_) 

    # DAFTAR UNIK UNTUK SELECTBOX CITY DAN PROFESSION
    # Daftar ini harus sesuai dengan output df_cleaned di create_pkl_files.py
    CITY_OPTIONS = ['Visakhapatnam', 'Bangalore', 'Srinagar', 'Varanasi', 'Jaipur', 'Pune', 'Kolkata', 
                    'Chennai', 'Mumbai', 'Delhi', 'Hyderabad', 'Ahmedabad', 'Surat', 'Lucknow', 'Other City'] 
    PROFESSION_OPTIONS = ['Student', 'Working Professional', 'Unemployed', 'Other Profession'] # Disesuaikan

    
except Exception as e:
    st.error(f"Terjadi kesalahan fatal saat memuat file PKL atau mendapatkan Feature Names: {e}")
    st.warning("Pastikan Anda sudah membuat ulang file PKL dengan skrip create_pkl_files.py terbaru.")
    st.stop()


# --- 2. MAPPINGS & DEFAULTS ---
SLEEP_MAP = {"Less than 5 hours": 1.0, "5-6 hours": 2.0, "7-8 hours": 3.0, "More than 8 hours": 4.0, "Others": 0.0}
FINANCIAL_MAP = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "?": 0.0}
SUICIDAL_MAP = {"No": 0.0, "Yes": 1.0}
FAMILY_MAP = {"No": 0.0, "Yes": 1.0}

FINANCIAL_OPTIONS = ["1", "2", "3", "4", "5", "?"] 
CITY_DEFAULT = "Visakhapatnam"
PROFESSION_DEFAULT = "Student"

# DAFTAR ANOMALI UNTUK CLEANING LIVE (Penting untuk Konsistensi)
ANOMALIES_CITY = [
    '3.0', 'Gaurav', 'Harsh', 'Harsha', 'Kibara', 'M.Com', 'M.Tech', 'ME',
    'Mihir', 'Mira', 'Nalini', 'Nandini', 'Rashi', 'Reyansh', 'Saanvi', 
    'Vaanya', 'Less Delhi', 'Less than 5 Kalyan'
]


# ==========================
# 🔧 PREPROCESSING & PREDICTION FUNCTION
# ==========================
def preprocess_and_predict(input_df):
    
    # 1. Cleaning Anomali (Sama seperti create_pkl_files.py)
    input_df['City'] = input_df['City'].astype(str).str.strip()
    
    # Logic: Jika input City termasuk anomali (ini hanya terjadi jika kita menggunakan text_input, tapi tetap dipertahankan untuk konsistensi cleaning)
    if input_df['City'].iloc[0] in ANOMALIES_CITY:
        input_df['City'] = 'Other City'
    
    input_df['Profession'] = input_df['Profession'].astype(str).str.strip()
    
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
    # Transformasi City dan Profession (menggunakan nilai dari selectbox)
    input_df[['City', 'Profession']] = te.transform(input_df[['City', 'Profession']].astype(str))
    
    # d. Scaling 
    input_scaled = scaler.transform(input_df[FEATURE_NAMES])

    # 3. Prediksi
    prediction = best_gb.predict(input_scaled)[0]
    prediction_proba = best_gb.predict_proba(input_scaled)[0]
    
    return prediction, prediction_proba

# --- 3. STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide")
st.title("Mental Health Predictor: UJI BOBOT SEMUA FITUR 🧠")
st.markdown("Model menggunakan Gradient Boosting.")
st.write("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.header("1. Demografi & Akademik")
    gender_input = st.selectbox("Gender", GENDER_OPTIONS)
    age = st.slider("Age (Usia)", 18, 60, 25)
    cgpa = st.slider("CGPA (Skala 0-10)", 0.0, 10.0, 7.5, 0.01)
    academic_pressure = st.slider("Academic Pressure", 0, 5, 3, step=1) 

with col_b:
    st.header("2. Gaya Hidup & Stres")
    work_study_hours = st.slider("Work/Study Hours (Jam/Hari)", 0, 12, 8, step=1)
    study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3, step=1)
    sleep_duration_input = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()))
    dietary_habits_input = st.selectbox("Dietary Habits", DIETARY_HABITS)


with col_c:
    st.header("3. Riwayat & Lainnya")
    degree_input = st.selectbox("Degree", DEGREES)
    # SELECTBOX CITY DAN PROFESSION
    city_input = st.selectbox("City", CITY_OPTIONS, index=CITY_OPTIONS.index(CITY_DEFAULT))
    profession_input = st.selectbox("Profession", PROFESSION_OPTIONS, index=PROFESSION_OPTIONS.index(PROFESSION_DEFAULT))
    
    financial_stress_input = st.selectbox("Financial Stress", FINANCIAL_OPTIONS) 
    suicidal_thoughts_input = st.selectbox("Pernah punya pikiran bunuh diri?", ["No", "Yes"]) 
    family_history_input = st.selectbox("Riwayat Keluarga Gangguan Mental", ["No", "Yes"])


st.write("---")

# --- 4. PREDICTION LOGIC ---

if st.button("PREDIKSI DAN UJI PENGARUH"):
    # 1. Kumpulkan data input
    data = {
        'Gender': [gender_input], 
        'Age': [age], 
        'City': [city_input], 
        'Profession': [profession_input],
        COL_ACADEMIC_PRESSURE: [float(academic_pressure)], 
        'CGPA': [cgpa], 
        COL_STUDY_SAT: [float(study_satisfaction)],
        COL_SLEEP: [sleep_duration_input], 
        'Dietary Habits': [dietary_habits_input], 
        'Degree': [degree_input],
        COL_SUICIDAL: [suicidal_thoughts_input], 
        COL_WORK_STUDY_HOURS: [float(work_study_hours)], 
        COL_FINANCIAL: [financial_stress_input], 
        COL_FAMILY: [family_history_input]
    }
    input_df = pd.DataFrame(data)
    
    # 2. Prediksi
    prediction, prediction_proba = preprocess_and_predict(input_df)
    
    # 3. Tampilkan Hasil
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
