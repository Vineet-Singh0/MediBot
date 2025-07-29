import pandas as pd

# --- Realistic disease-symptom clusters (edit/expand as needed) ---
disease_clusters = {
    'common cold': ['fever', 'headache', 'runny_nose', 'sneezing', 'sore_throat', 'cough', 'fatigue'],
    'flu': ['fever', 'headache', 'fatigue', 'muscle_pain', 'cough', 'sore_throat', 'chills'],
    'allergy': ['sneezing', 'runny_nose', 'nasal_congestion', 'watery_eyes', 'itching'],
    'migraine': ['headache', 'nausea', 'vomiting', 'sensitivity_to_light', 'throbbing_pain'],
    'gastritis': ['abdominal_pain', 'nausea', 'vomiting', 'loss_of_appetite', 'bloating'],
    'urinary tract infection': ['burning_micturition', 'frequent_urination', 'lower_abdominal_pain', 'fever'],
    'anemia': ['fatigue', 'pallor', 'shortness_of_breath', 'dizziness'],
    'gerd': ['heartburn', 'regurgitation', 'chest_pain', 'sore_throat'],
    'viral fever': ['fever', 'headache', 'body_ache', 'fatigue'],
    'tension headache': ['headache', 'neck_pain', 'fatigue'],
    'conjunctivitis': ['red_eyes', 'itching', 'watering_from_eyes', 'discharge'],
    'sinusitis': ['headache', 'facial_pain', 'nasal_congestion', 'runny_nose'],
    'bronchitis': ['cough', 'sputum', 'chest_discomfort', 'fatigue'],
    'typhoid': ['fever', 'abdominal_pain', 'headache', 'constipation', 'fatigue'],
    'chicken pox': ['fever', 'itching', 'skin_rash', 'fatigue'],
    'hypertension': ['headache', 'dizziness', 'chest_pain', 'blurred_vision'],
    'diabetes': ['excessive_thirst', 'frequent_urination', 'fatigue', 'weight_loss'],
    'asthma': ['shortness_of_breath', 'wheezing', 'chest_tightness', 'cough'],
    'jaundice': ['yellowish_skin', 'dark_urine', 'fatigue', 'abdominal_pain'],
    'appendicitis': ['abdominal_pain', 'nausea', 'vomiting', 'loss_of_appetite'],
    'pneumonia': ['fever', 'cough', 'shortness_of_breath', 'chest_pain'],
    'arthritis': ['joint_pain', 'swelling', 'stiffness', 'fatigue'],
    'hypothyroidism': ['fatigue', 'weight_gain', 'cold_intolerance', 'dry_skin'],
    'hyperthyroidism': ['weight_loss', 'palpitations', 'heat_intolerance', 'tremor'],
    'peptic ulcer disease': ['abdominal_pain', 'nausea', 'vomiting', 'bloating'],
    'dengue': ['fever', 'headache', 'muscle_pain', 'rash'],
    'malaria': ['fever', 'chills', 'sweating', 'headache'],
    'tuberculosis': ['cough', 'weight_loss', 'fever', 'night_sweats'],
    'meningitis': ['fever', 'headache', 'neck_stiffness', 'photophobia', 'nausea'],
    'hepatitis a': ['fever', 'fatigue', 'loss_of_appetite', 'jaundice'],
    'hepatitis b': ['fatigue', 'jaundice', 'abdominal_pain', 'joint_pain'],
    'hepatitis c': ['fatigue', 'jaundice', 'muscle_pain', 'joint_pain'],
    'hepatitis d': ['jaundice', 'abdominal_pain', 'fatigue', 'nausea'],
    'hepatitis e': ['jaundice', 'fatigue', 'abdominal_pain', 'nausea'],
    'aids': ['weight_loss', 'fever', 'night_sweats', 'fatigue'],
    'heart attack': ['chest_pain', 'shortness_of_breath', 'sweating', 'nausea'],
    'psoriasis': ['skin_rash', 'itching', 'red_patches', 'scaling'],
    'impetigo': ['skin_rash', 'blister', 'itching', 'crusting'],
    'cervical spondylosis': ['neck_pain', 'stiffness', 'headache', 'shoulder_pain'],
    'varicose veins': ['leg_pain', 'swelling', 'heaviness', 'itching'],
    'dimorphic hemmorhoids (piles)': ['rectal_pain', 'bleeding', 'itching', 'swelling'],
    'chronic cholestasis': ['itching', 'jaundice', 'dark_urine', 'fatigue'],
    'alcoholic hepatitis': ['jaundice', 'abdominal_pain', 'fatigue', 'nausea'],
    # Add more as needed
}

# Fix: Use correct path for precaution file
precaution_df = pd.read_csv('data/symptom_precaution.csv')
precaution_df.columns = [col.strip().lower().replace(" ", "_") for col in precaution_df.columns]
precaution_col = 'disease' if 'disease' in precaution_df.columns else 'prognosis'

# Doctor mapping (based on real-world guidelines)
doctor_mapping = {
    'heart attack': 'Cardiologist',
    'hypertension': 'Cardiologist',
    'asthma': 'Pulmonologist',
    'bronchitis': 'Pulmonologist',
    'pneumonia': 'Pulmonologist',
    'tuberculosis': 'Pulmonologist',
    'diabetes': 'Endocrinologist',
    'hypothyroidism': 'Endocrinologist',
    'psoriasis': 'Dermatologist',
    'fungal infection': 'Dermatologist',
    'allergy': 'Allergist/Immunologist',
    'arthritis': 'Rheumatologist',
    'migraine': 'Neurologist',
    'paralysis (brain hemorrhage)': 'Neurologist',
    'jaundice': 'Gastroenterologist',
    'hepatitis a': 'Gastroenterologist',
    'hepatitis b': 'Gastroenterologist',
    'hepatitis c': 'Gastroenterologist',
    'hepatitis d': 'Gastroenterologist',
    'hepatitis e': 'Gastroenterologist',
    'alcoholic hepatitis': 'Gastroenterologist',
    'typhoid': 'General Physician',
    'dengue': 'General Physician',
    'malaria': 'General Physician',
    'urinary tract infection': 'Urologist',
    'aids': 'Infectious Disease',
    'cancer': 'Oncologist',
    'dimorphic hemmorhoids (piles)': 'Gastroenterologist',
    'drug reaction': 'Dermatologist',
    'peptic ulcer disease': 'Gastroenterologist',
    # Add more disease-specific mappings as needed
}

rows = []
for disease, symptoms in disease_clusters.items():
    disease_lc = disease.lower()
    precaution_row = precaution_df[precaution_df[precaution_col].str.lower() == disease_lc]
    precautions = []
    if not precaution_row.empty:
        for c in precaution_df.columns:
            if 'precaution' in c and pd.notna(precaution_row.iloc[0][c]) and str(precaution_row.iloc[0][c]).strip() != '':
                precautions.append(str(precaution_row.iloc[0][c]))
    doctor_type = doctor_mapping.get(disease_lc, 'General Physician')
    rows.append({
        'disease': disease.title(),
        'symptom_list': ', '.join(sorted(set(symptoms))),
        'doctor_type': doctor_type,
        'precaution': ', '.join(precautions) if precautions else "Consult a doctor"
    })

# Ensure correct column order
cols = ['disease', 'symptom_list', 'doctor_type', 'precaution']
df = pd.DataFrame(rows)[cols]
df.to_csv('outputs/disease_centric.csv', index=False)
print("Disease-centric dataset ready: disease_centric_knowledgebase_with_doctor.csv")



