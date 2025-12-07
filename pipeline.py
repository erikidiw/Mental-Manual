# pipeline.py

# --- DEFENISI BOBOT MANUAL ---
# Bobot ini didasarkan pada visualisasi awal Anda (Suicidal=tertinggi, Academic>Financial, dll.)
BOBOT_SUICIDE = 15
BOBOT_ACADEMIC_PRESSURE = 4
BOBOT_FINANCIAL_STRESS = 3
BOBOT_STUDY_HOURS = 1
BOBOT_SLEEP_DURATION = 1
BOBOT_DIETARY = 1
BOBOT_FAMILY_HISTORY = 5
BOBOT_CITY = 2 # Bobot untuk City yang bukan 'Lainnya'

# MAPPING 
# Mengubah kategori durasi tidur dan diet menjadi skor risiko numerik
MAP_SLEEP_RISK = {'Less than 5 hours': 3, '5-6 hours': 2, '7-8 hours': 0, 'More than 8 hours': 0, 'Others': 3} 
MAP_DIETARY_RISK = {'Tidak Sehat': 1, 'Sehat': 0}

def hitung_skor_risiko(data_input):
    # Menghitung total skor risiko depresi (Manual Scoring Komprehensif).
    skor = 0
    
    # 1. Have you ever had suicidal thoughts? (Bobot 15)
    if data_input.get('suicide') == 'Yes':
        skor += BOBOT_SUICIDE
    
    # 2. Academic Pressure (Skala 1-5, Bobot 4)
    academic_score = data_input.get('academic', 0)
    skor += academic_score * BOBOT_ACADEMIC_PRESSURE

    # 3. Financial Stress (Skala 1-5, Bobot 3)
    financial_score = data_input.get('financial', 0)
    skor += financial_score * BOBOT_FINANCIAL_STRESS
    
    # 4. Study Hours (Bobot 1, risiko jika jam > 8)
    if data_input.get('hours', 0) > 8:
        skor += BOBOT_STUDY_HOURS
    
    # 5. Sleep Duration (Bobot 1)
    sleep_category = data_input.get('sleep')
    sleep_risk_score = MAP_SLEEP_RISK.get(sleep_category, 0)
    skor += sleep_risk_score * BOBOT_SLEEP_DURATION

    # 6. Dietary Habits (Bobot 1)
    if data_input.get('diet') == 'Tidak Sehat':
        skor += BOBOT_DIETARY
    
    # 7. CGPA (Bobot Struktural Tinggi)
    # Logika berdasarkan batas risiko dari arsitektur ML lama (<3.0 = highest, <5.0 = medium)
    cgpa = data_input.get('cgpa', 10)
    if cgpa < 3.0: 
        skor += 15
    elif cgpa < 5.0: 
        skor += 8
        
    # 8. Study Satisfaction (Bobot Struktural Tinggi)
    satisfaction = data_input.get('satisfaction', 5)
    if satisfaction <= 2: 
        skor += 10
        
    # 9. Family History (Bobot 5)
    if data_input.get('history') == 'Yes':
        skor += BOBOT_FAMILY_HISTORY
        
    # 10. City (Bobot 2)
    city_input = data_input.get('city')
    # Memberikan skor jika City bukan kategori umum/kecil
    if city_input not in ['Lainnya', 'Others', 'Unknown']:
        skor += BOBOT_CITY
        
    return skor

def tetapkan_level_risiko(skor_total):
    # Menetapkan level risiko berdasarkan skor total. (Skor Maksimum Total ~ 83)
    if skor_total > 50: 
        return {
            'level': "RISIKO TINGGI",
            'pesan': "Risiko sangat tinggi. Konsultasi segera disarankan.",
            'warna': 'red'
        }
    elif skor_total > 25:
        return {
            'level': "RISIKO SEDANG",
            'pesan': "Faktor risiko perlu diwaspadai. Perbaiki manajemen stres dan pola hidup.",
            'warna': 'orange'
        }
    else:
        return {
            'level': "RISIKO RENDAH",
            'pesan': "Risiko rendah. Pertahankan pola hidup seimbang.",
            'warna': 'green'
        }
