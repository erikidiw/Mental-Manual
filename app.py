import streamlit as st
import pandas as pd
from pipeline import hitung_skor_risiko, tetapkan_level_risiko
from category_encoders.target_encoder import TargetEncoder

# ==========================
# 🔧 CUSTOM SCORING LOGIC (MANUAL SCORING - DITAMBAHKAN)
# ==========================

# --- DEFENISI BOBOT MANUAL (Sesuai kesepakatan akhir) ---
BOBOT_SUICIDE = 15
BOBOT_ACADEMIC_PRESSURE = 4
BOBOT_FINANCIAL_STRESS = 3
BOBOT_STUDY_HOURS = 1
BOBOT_SLEEP_DURATION = 1
BOBOT_DIETARY = 1
BOBOT_FAMILY_HISTORY = 5

# MAPPING 
MAP_SLEEP_RISK = {'Less than 5 hours': 3, '5-6 hours': 2, '7-8 hours': 0, 'More than 8 hours': 0, 'Others': 3} 
MAP_DIETARY_RISK = {'Tidak Sehat': 1, 'Sehat': 0}

def hitung_skor_risiko(data_input):
    """Menghitung total skor risiko depresi menggunakan Bobot Manual (Bukan ML)."""
    skor = 0
    
    # 1. Have you ever had suicidal thoughts? (Bobot 15)
    if data_input.get('suicide') == 'Yes':
        skor += BOBOT_SUICIDE
    
    # 2. Academic Pressure (Skala 1-5, Bobot 4)
    skor += data_input.get('academic', 0) * BOBOT_ACADEMIC_PRESSURE

    # 3. Financial Stress (Skala 1-5, Bobot 3)
    skor += data_input.get('financial', 0) * BOBOT_FINANCIAL_STRESS
    
    # 4. Study Hours (Bobot 1, risiko jika jam > 8)
    if data_input.get('hours', 0) > 8:
        skor += BOBOT_STUDY_HOURS
    
    # 5. Sleep Duration (Bobot 1)
    sleep_risk_score = MAP_SLEEP_RISK.get(data_input.get('sleep'), 0)
    skor += sleep_risk_score * BOBOT_SLEEP_DURATION

    # 6. Dietary Habits (Bobot 1)
    # Catatan: Kita perlu memetakan 'Dietary Habits' dari input ML ('Sehat'/'Tidak Sehat') ke MAP_DIETARY_RISK
    if data_input.get('diet') == 'Tidak Sehat': # Asumsi input dari selectbox
        skor += BOBOT_DIETARY
    
    # 7. CGPA (Bobot 15 / 8)
    cgpa = data_input.get('cgpa', 10)
    if cgpa < 3.0: 
        skor += 15
    elif cgpa < 5.0: 
        skor += 8
        
    # 8. Study Satisfaction (Bobot 10)
    satisfaction = data_input.get('satisfaction', 5)
    if satisfaction <= 2: 
        skor += 10
        
    # 9. Family History (Bobot 5)
    if data_input.get('history') == 'Yes':
        skor += BOBOT_FAMILY_HISTORY
        
    # Catatan: City, Gender, Profession, Degree tidak memiliki bobot skor manual > 0
    
    return skor

def tetapkan_level_risiko(skor_total):
    """Menetapkan level risiko berdasarkan skor total manual."""
    if skor_total > 50: 
        return { 'level': "RISIKO TINGGI", 'warna': 'red' }
    elif skor_total > 25:
        return { 'level': "RISIKO SEDANG", 'warna': 'orange' }
    else:
        return { 'level': "RISIKO RENDAH", 'warna': 'green' }


# ==========================
# 🔧 Custom Preprocessing Classes (ML Code)
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
# 🚀 Load Artifacts (ML Code)
# ==========================
try:
    # Memuat ML artifacts hanya untuk mendapatkan UNIQUE_OPTS
    artifacts = joblib.load('pipeline_artifacts.pkl') 
    # Tetap memuat pipeline dan encoders meskipun tidak digunakan untuk prediksi
    pipeline = artifacts['pipeline'] 
    label_encoders = artifacts['label_encoders']
    target_encoder = artifacts['target_encoder']
    
    ordinal_mapping_data = artifacts['ordinal_mapper'].mappings 
    ordinal_mapper = CustomOrdinalMapper(ordinal_mapping_data) 
    
    feature_cols = artifacts['feature_cols']
    UNIQUE_OPTS = artifacts['unique_options']
    
    st.success("Konfigurasi ML dimuat (Hanya untuk opsi Selectbox).")
    
except Exception as e:
    # Jika PKL gagal dimuat, kita perlu menghentikan aplikasi atau menggunakan opsi default
    st.error(f"Gagal memuat artifacts. Pastikan 'pipeline_artifacts.pkl' sudah dibuat: {e}")
    st.stop()


# ==========================
# 🔧 PREPROCESSING & PREDICTION FUNCTION (ML Code - Tidak Dipakai)
# ==========================

def preprocess_and_predict(input_data):
    # Fungsi ini tidak dipanggil, tetapi dipertahankan agar kode tetap utuh
    return 0 # Mengembalikan nilai dummy

# ==========================
# 🧠 STREAMLIT UI (Menggunakan Opsi dari PKL)
# ==========================

st.title("Sistem Prediksi Risiko Depresi Mahasiswa")
st.write("Skor risiko dihitung berdasarkan Bobot Manual.")

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
    # Menggunakan kunci 'Dietary Habits' untuk Selectbox
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
if st.button("Hitung Skor Risiko"):
    
    cgpa_actual = gpa_input * 2.5
    
    # PERHATIAN: Memastikan nilai Financial Stress dikonversi ke integer (1-5)
    # Ini berbeda dari input ML lama ('3.0'), tetapi diperlukan untuk scoring manual (x3)
    financial_int = int(financial) 
    
    # Input data disusun untuk fungsi scoring manual
    input_data_scoring = {
        "academic": academic,
        "financial": financial_int,
        "hours": hours,
        "sleep": sleep,
        "diet": diet,
        "suicide": suicide,
        "cgpa": cgpa_actual,         
        "satisfaction": satisfaction, 
        "history": history, 
        "city": city, # City ditambahkan ke input untuk scoring (walaupun bobotnya 2)
    }

    # Menggunakan fungsi SCORING MANUAL, bukan fungsi ML (preprocess_and_predict)
    skor_total = hitung_skor_risiko(input_data_scoring)
    hasil = tetapkan_level_risiko(skor_total)

    st.subheader("Hasil Penilaian Risiko")
    st.markdown(f"Skor Total Risiko: {skor_total}")
    
    if hasil['warna'] == 'red':
        st.error(f"⚠️ {hasil['level']}")
    elif hasil['warna'] == 'orange':
        st.error(f"🔥 {hasil['level']}")
    else:
        st.success(f"✅ {hasil['level']}")
        
    st.info(f"Rekomendasi: {hasil['pesan']}")
