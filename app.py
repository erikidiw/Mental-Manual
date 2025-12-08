import streamlit as st
import pandas as pd
import joblib
from category_encoders.target_encoder import TargetEncoder
import numpy as np 

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


# ==========================
# 🚀 Load Artifacts
# ==========================
try:
    artifacts = joblib.load('pipeline_artifacts.pkl') 
    pipeline = artifacts['pipeline']
    label_encoders = artifacts['label_encoders']
    target_encoder = artifacts['target_encoder']
    
    ordinal_mapping_data = artifacts['ordinal_mapper'].mappings 
    ordinal_mapper = CustomOrdinalMapper(ordinal_mapping_data) 
    
    feature_cols = artifacts['feature_cols']
    UNIQUE_OPTS = artifacts['unique_options']
    
    st.success("Model Ensemble (Voting Classifier) berhasil dimuat.")
    
except Exception as e:
    st.error(f"Gagal memuat artifacts. Pastikan 'pipeline_artifacts.pkl' sudah dibuat: {e}")
    st.stop()


# ==========================
# 🔧 PREPROCESSING & PREDICTION FUNCTION
# ==========================

def preprocess_and_predict(input_data):
    df_single = pd.DataFrame([input_data])
    df_single = df_single[feature_cols]

    # Cleaning Dasar
    for col in ['Sleep Duration', 'Financial Stress']:
        if col in df_single.columns:
            df_single[col] = df_single[col].astype(str).str.replace("'", "").str.strip()
            
    df_single['Financial Stress'] = df_single['Financial Stress'].replace('?', '0')

    # Ordinal Mapping
    df_single_ordinal = ordinal_mapper.transform(df_single)
    for col in ordinal_mapper.cols:
        df_single[col] = df_single_ordinal[col]
    
    # Label Encoding
    label_cols = list(label_encoders.keys())
    for col in label_cols:
        if col in df_single.columns:
            df_single[col] = df_single[col].astype(str)
            le = label_encoders[col]
            def encode_label(val):
                if val in le.classes_:
                    return le.transform([val])[0]
                return -1 
            df_single[col] = df_single[col].apply(encode_label)
    
    # Target Encoding
    target_cols = ['City', 'Profession']
    df_single[target_cols] = target_encoder.transform(df_single[target_cols])
    
    # Konversi ke float
    df_processed = df_single.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    prediction = pipeline.predict(df_processed)[0]
    return prediction


# ==========================
# 🧠 STREAMLIT UI
# ==========================

st.title("Sistem Prediksi Risiko Depresi Mahasiswa (Ensemble Model)")
st.write("Skor risiko ditentukan sepenuhnya oleh bobot yang dipelajari model.")

col1, col2, col3 = st.columns(3)

profession_options = UNIQUE_OPTS['Profession'] + ["Others"]

with col1:
    st.subheader("Informasi Dasar")
    gender = st.selectbox("Jenis Kelamin", UNIQUE_OPTS['Gender'])
    city = st.selectbox("Kota Tinggal", UNIQUE_OPTS['City'])
    profession = st.selectbox("Pekerjaan", profession_options) 
    age = st.number_input("Umur", min_value=10, max_value=80, value=25, step=1)
    degree = st.selectbox("Jenjang Pendidikan (Degree)", UNIQUE_OPTS['Degree'])
    
with col2:
    st.subheader("Faktor Akademik dan Kehidupan")
    gpa_input = st.number_input("Rata-rata Nilai (GPA)", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    hours = st.number_input("Jam Belajar/Kerja per hari", min_value=0, max_value=20, value=5, step=1)
    sleep = st.selectbox("Durasi Tidur", UNIQUE_OPTS['Sleep Duration'])
    diet = st.selectbox("Kebiasaan Makan (Dietary Habits)", UNIQUE_OPTS['Dietary Habits'])
    
with col3:
    st.subheader("Faktor Risiko Mental")
    academic = st.slider("Tekanan Akademik (1=Rendah, 5=Tinggi)", min_value=1, max_value=5, value=3, step=1)
    satisfaction = st.slider("Kepuasan Belajar (1=Rendah, 5=Tinggi)", min_value=1, max_value=5, value=4, step=1)
    financial = st.slider("Stres Keuangan (1=Rendah, 5=Tinggi)", min_value=1, max_value=5, value=3, step=1)
    
    history = st.selectbox("Riwayat Mental Keluarga", UNIQUE_OPTS['Family History'])
    suicide = st.selectbox("Pernah terpikir Bunuh Diri?", UNIQUE_OPTS['Suicidal Thoughts']) 


# Tombol Prediksi
st.markdown("---")
if st.button("Prediksi Tingkat Risiko"):
    
    cgpa_actual = gpa_input * 2.5
    
    input_data = {
        "Gender": gender,
        "City": city,
        "Profession": profession,
        "Age": age,
        "CGPA": cgpa_actual, 
        "Work/Study Hours": hours,
        "Sleep Duration": sleep,
        "Dietary Habits": diet,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicide,
        "Financial Stress": str(financial) + ".0", 
        "Family History of Mental Illness": history,
        "Academic Pressure": academic,
        "Study Satisfaction": satisfaction,
    }

    prediction = preprocess_and_predict(input_data)

    st.subheader("Hasil Prediksi")
    
    # --- TANPA SAFETY OVERRIDE: MURNI PREDIKSI MODEL ---
    if prediction >= 1: 
        st.error("🔥 **POTENSI/RISIKO DEPRESI**")
        if prediction == 2:
            st.warning("Risiko sangat tinggi menurut model. Konsultasi dan tindakan segera disarankan.")
        else:
            st.info("Risiko sedang menurut model. Disarankan mencari dukungan konseling.")
            
    else: # prediction == 0 
        st.success("✅ **TIDAK DEPRESI**")
        st.info("Risiko rendah menurut model. Pertahankan pola hidup seimbang.")
