import pandas as pd
import random
import numpy as np
from tqdm import tqdm

# 1. Load Files
symptom_df = pd.read_csv('data/dataset.csv')
precaution_df = pd.read_csv('data/symptom_precaution.csv')
doctor_df = pd.read_csv('data/disease_centric_knowledgebase_with_doctor.csv')

# 2. Clean Columns
symptom_df.columns = [col.strip().lower().replace(" ", "_") for col in symptom_df.columns]
precaution_df.columns = [col.strip().lower().replace(" ", "_") for col in precaution_df.columns]
doctor_df.columns = [col.strip().lower().replace(" ", "_") for col in doctor_df.columns]

# 3. Identify Columns
disease_col = 'prognosis' if 'prognosis' in symptom_df.columns else 'disease'
precaution_col = 'disease' if 'disease' in precaution_df.columns else 'prognosis'
symptom_cols = [col for col in symptom_df.columns if 'symptom' in col]

# 4. Unique Symptoms (clean up empty, nan)
all_symptoms = sorted(set(
    str(val).strip().lower().replace(" ", "_")
    for col in symptom_cols
    for val in symptom_df[col]
    if pd.notna(val) and str(val).strip() != '' and str(val).strip().lower() != 'nan'
))

# 5. Universal Symptoms (for realistic core/optional split)
universal_symptoms = {'fever', 'fatigue', 'headache'}  # <-- Adjust as per your dataset

# 6. Mapping: Disease → Symptoms (list)
disease_symptom_map = {}
for _, row in symptom_df.iterrows():
    disease = row[disease_col].strip().lower()
    symptoms = [str(row[c]).strip().lower().replace(' ', '_') for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
    if disease not in disease_symptom_map:
        disease_symptom_map[disease] = []
    disease_symptom_map[disease].extend([s for s in symptoms if s not in disease_symptom_map[disease]])

# 6.1 Symptom Clusters for Common Diseases
# These clusters represent classic presentations for key diseases
# You can expand this dictionary as needed

disease_clusters = {
    # --- Common Diseases ---
    'common cold': [
        {'fever', 'headache', 'runny_nose', 'sneezing', 'sore_throat', 'cough', 'fatigue'},
        {'runny_nose', 'sneezing', 'sore_throat'},
        {'fever', 'runny_nose', 'sneezing'}
    ],
    'flu': [
        {'fever', 'headache', 'fatigue', 'muscle_pain', 'cough', 'sore_throat', 'chills'},
        {'fever', 'cough', 'sore_throat', 'fatigue'}
    ],
    'allergy': [
        {'sneezing', 'runny_nose', 'nasal_congestion', 'watery_eyes', 'itching'},
        {'sneezing', 'itching', 'rash'}
    ],
    'migraine': [
        {'headache', 'nausea', 'vomiting', 'sensitivity_to_light', 'throbbing_pain'},
        {'headache', 'visual_disturbance', 'throbbing_pain'}
    ],
    'gastritis': [
        {'abdominal_pain', 'nausea', 'vomiting', 'loss_of_appetite', 'bloating'},
        {'abdominal_pain', 'indigestion', 'heartburn'}
    ],
    'urinary tract infection': [
        {'burning_micturition', 'frequent_urination', 'lower_abdominal_pain', 'fever'},
        {'burning_micturition', 'cloudy_urine', 'urgency'}
    ],
    'anemia': [
        {'fatigue', 'pallor', 'shortness_of_breath', 'dizziness'},
        {'weakness', 'palpitations', 'headache'}
    ],
    'gerd': [
        {'heartburn', 'regurgitation', 'chest_pain', 'sore_throat'},
        {'acid_reflux', 'indigestion', 'cough'}
    ],
    'viral fever': [
        {'fever', 'headache', 'body_ache', 'fatigue'},
        {'fever', 'chills', 'sore_throat'}
    ],
    'tension headache': [
        {'headache', 'neck_pain', 'fatigue'},
        {'headache', 'difficulty_concentrating'}
    ],
    'conjunctivitis': [
        {'red_eyes', 'itching', 'watering_from_eyes', 'discharge'},
        {'red_eyes', 'swelling_eyelid', 'pain_in_eyes'}
    ],
    'sinusitis': [
        {'headache', 'facial_pain', 'nasal_congestion', 'runny_nose'},
        {'postnasal_drip', 'cough', 'fever'}
    ],
    'bronchitis': [
        {'cough', 'sputum', 'chest_discomfort', 'fatigue'},
        {'cough', 'wheezing', 'shortness_of_breath'}
    ],
    # --- Medium Prevalence Diseases ---
    'typhoid': [
        {'fever', 'abdominal_pain', 'headache', 'constipation', 'fatigue'},
        {'fever', 'diarrhoea', 'loss_of_appetite', 'nausea'}
    ],
    'chicken pox': [
        {'fever', 'itching', 'skin_rash', 'fatigue'},
        {'skin_rash', 'itching', 'blister', 'fatigue'}
    ],
    'hypertension': [
        {'headache', 'dizziness', 'chest_pain', 'blurred_vision'},
        {'headache', 'shortness_of_breath', 'fatigue'}
    ],
    'diabetes': [
        {'excessive_thirst', 'frequent_urination', 'fatigue', 'weight_loss'},
        {'blurred_vision', 'slow_healing_sores', 'hunger'}
    ],
    'asthma': [
        {'shortness_of_breath', 'wheezing', 'chest_tightness', 'cough'},
        {'cough', 'wheezing', 'difficulty_breathing'}
    ],
    'jaundice': [
        {'yellowish_skin', 'dark_urine', 'fatigue', 'abdominal_pain'},
        {'yellowish_skin', 'loss_of_appetite', 'itching'}
    ],
    'appendicitis': [
        {'abdominal_pain', 'nausea', 'vomiting', 'loss_of_appetite'},
        {'abdominal_pain', 'fever', 'constipation'}
    ],
    'pneumonia': [
        {'fever', 'cough', 'shortness_of_breath', 'chest_pain'},
        {'cough', 'fatigue', 'sputum'}
    ],
    'arthritis': [
        {'joint_pain', 'swelling', 'stiffness', 'fatigue'},
        {'joint_pain', 'limited_movement', 'redness'}
    ],
    'hypothyroidism': [
        {'fatigue', 'weight_gain', 'cold_intolerance', 'dry_skin'},
        {'constipation', 'hair_loss', 'depression'}
    ],
    'hyperthyroidism': [
        {'weight_loss', 'palpitations', 'heat_intolerance', 'tremor'},
        {'anxiety', 'sweating', 'diarrhoea'}
    ],
    'peptic ulcer diseae': [
        {'abdominal_pain', 'nausea', 'vomiting', 'bloating'},
        {'heartburn', 'loss_of_appetite', 'weight_loss'}
    ],
    'dengue': [
        {'fever', 'headache', 'muscle_pain', 'rash'},
        {'fever', 'pain_behind_eyes', 'joint_pain'}
    ],
    'malaria': [
        {'fever', 'chills', 'sweating', 'headache'},
        {'fever', 'nausea', 'vomiting', 'headache'}
    ],
    'tuberculosis': [
        {'cough', 'weight_loss', 'fever', 'night_sweats'},
        {'cough', 'chest_pain', 'fatigue'}
    ],
    # --- Rare Diseases ---
    'meningitis': [
        {'fever', 'headache', 'neck_stiffness', 'photophobia', 'nausea'},
        {'vomiting', 'confusion', 'seizures'}
    ],
    'hepatitis a': [
        {'fever', 'fatigue', 'loss_of_appetite', 'jaundice'},
        {'dark_urine', 'abdominal_pain', 'nausea'}
    ],
    'hepatitis b': [
        {'fatigue', 'jaundice', 'abdominal_pain', 'joint_pain'},
        {'dark_urine', 'nausea', 'vomiting'}
    ],
    'hepatitis c': [
        {'fatigue', 'jaundice', 'muscle_pain', 'joint_pain'},
        {'abdominal_pain', 'itching', 'nausea'}
    ],
    'hepatitis d': [
        {'jaundice', 'abdominal_pain', 'fatigue', 'nausea'},
        {'dark_urine', 'vomiting', 'loss_of_appetite'}
    ],
    'hepatitis e': [
        {'jaundice', 'fatigue', 'abdominal_pain', 'nausea'},
        {'dark_urine', 'vomiting', 'itching'}
    ],
    'aids': [
        {'weight_loss', 'fever', 'night_sweats', 'fatigue'},
        {'recurrent_infections', 'diarrhoea', 'skin_rash'}
    ],
    'heart attack': [
        {'chest_pain', 'shortness_of_breath', 'sweating', 'nausea'},
        {'fatigue', 'palpitations', 'anxiety'}
    ],
    'psoriasis': [
        {'skin_rash', 'itching', 'red_patches', 'scaling'},
        {'joint_pain', 'nail_changes', 'dry_skin'}
    ],
    'impetigo': [
        {'skin_rash', 'blister', 'itching', 'crusting'},
        {'redness', 'swelling', 'pain'}
    ],
    'cervical spondylosis': [
        {'neck_pain', 'stiffness', 'headache', 'shoulder_pain'},
        {'dizziness', 'numbness', 'tingling'}
    ],
    'varicose veins': [
        {'leg_pain', 'swelling', 'heaviness', 'itching'},
        {'skin_discoloration', 'ulcer', 'cramping'}
    ],
    'dimorphic hemmorhoids(piles)': [
        {'rectal_pain', 'bleeding', 'itching', 'swelling'},
        {'discomfort', 'mucus_discharge', 'constipation'}
    ],
    'chronic cholestasis': [
        {'itching', 'jaundice', 'dark_urine', 'fatigue'},
        {'abdominal_pain', 'loss_of_appetite', 'nausea'}
    ],
    'alcoholic hepatitis': [
        {'jaundice', 'abdominal_pain', 'fatigue', 'nausea'},
        {'vomiting', 'loss_of_appetite', 'swelling'}
    ],
    # Add more as needed
}

# 6.2 Age-specific Symptom Clusters (for selected diseases)
age_specific_clusters = {
    'pneumonia': {
        'child': [
            {'fever', 'cough', 'rapid_breathing', 'chest_indrawing'},
            {'fever', 'cough', 'vomiting', 'irritability'}
        ],
        'adult': [
            {'fever', 'cough', 'shortness_of_breath', 'chest_pain'},
            {'cough', 'fatigue', 'sputum'}
        ],
        'elderly': [
            {'confusion', 'weakness', 'loss_of_appetite', 'cough'},
            {'shortness_of_breath', 'fatigue', 'low_grade_fever'}
        ]
    },
    'urinary tract infection': {
        'child': [
            {'fever', 'vomiting', 'irritability', 'poor_feeding'},
            {'fever', 'abdominal_pain', 'enuresis'}
        ],
        'adult': [
            {'burning_micturition', 'frequent_urination', 'lower_abdominal_pain', 'fever'},
            {'burning_micturition', 'cloudy_urine', 'urgency'}
        ],
        'elderly': [
            {'confusion', 'weakness', 'incontinence', 'no_fever'},
            {'fatigue', 'urinary_urgency', 'loss_of_appetite'}
        ]
    },
    'chicken pox': {
        'child': [
            {'fever', 'itching', 'skin_rash', 'fatigue'},
            {'skin_rash', 'itching', 'blister', 'fatigue'}
        ],
        'adult': [
            {'fever', 'skin_rash', 'fatigue', 'headache'},
            {'skin_rash', 'blister', 'muscle_pain'}
        ],
        'elderly': [
            {'fever', 'skin_rash', 'fatigue', 'complications'},
            {'skin_rash', 'blister', 'weakness'}
        ]
    },
    'appendicitis': {
        'child': [
            {'abdominal_pain', 'vomiting', 'fever', 'irritability'},
            {'abdominal_pain', 'loss_of_appetite', 'nausea'}
        ],
        'adult': [
            {'abdominal_pain', 'nausea', 'vomiting', 'loss_of_appetite'},
            {'abdominal_pain', 'fever', 'constipation'}
        ],
        'elderly': [
            {'abdominal_pain', 'mild_fever', 'confusion'},
            {'abdominal_pain', 'nausea', 'weakness'}
        ]
    },
    'diabetes': {
        'child': [
            {'excessive_thirst', 'frequent_urination', 'weight_loss', 'fatigue'},
            {'bedwetting', 'abdominal_pain', 'vomiting'}
        ],
        'adult': [
            {'excessive_thirst', 'frequent_urination', 'fatigue', 'weight_loss'},
            {'blurred_vision', 'slow_healing_sores', 'hunger'}
        ],
        'elderly': [
            {'fatigue', 'weight_loss', 'confusion', 'incontinence'},
            {'blurred_vision', 'slow_healing_sores', 'weakness'}
        ]
    }
}

# 7. Mapping: Disease → Precaution
precaution_map = {}
for _, row in precaution_df.iterrows():
    dname = row[precaution_col].strip().lower()
    precs = [str(row[c]).strip() for c in precaution_df.columns if 'precaution' in c and pd.notna(row[c]) and str(row[c]).strip() != '']
    precaution_map[dname] = ', '.join(precs) if precs else "Consult a doctor"

# 8. Mapping: Disease → Doctor Type
doctor_df['disease'] = doctor_df['disease'].astype(str).str.strip().str.lower()
doctor_map = doctor_df.set_index('disease')['doctor_type'].to_dict()

# 9. Custom Weights (rare/mid class boost; adjust as per disease distribution!)
custom_weights = {
    'diabetes': 30, 'common cold': 28, 'migraine': 25,
    'allergy': 20, 'hypertension': 18, 'arthritis': 18, 'typhoid': 15,
    'tuberculosis': 12, 'pneumonia': 12, 'dengue': 10, 'chicken pox': 10,
    'hypothyroidism': 8, 'heart attack': 8, 'aids': 8, 'impetigo': 7, 'psoriasis': 7,
    'hepatitis a': 7, 'hepatitis b': 7, 'hepatitis c': 7, 'hepatitis d': 7, 'hepatitis e': 7,
    'gastroenteritis': 7, 'hypoglycemia': 7, 'cervical spondylosis': 7,
    'alcoholic hepatitis': 7, 'malaria': 7, 'chronic cholestasis': 7, 'jaundice': 7,
    '(vertigo) paroymsal  positional vertigo': 7, 'dimorphic hemmorhoids(piles)': 7,
    'bronchial asthma': 7, 'varicose veins': 7, 'acne': 7, 'osteoarthristis': 7,
    'hyperthyroidism': 7, 'paralysis (brain hemorrhage)': 7, 'peptic ulcer diseae': 7,
    'gerd': 7, 'urinary tract infection': 7, 'fungal infection': 7, 'drug reaction': 7
}
default_weight = 2
valid_diseases = list(disease_symptom_map.keys())
disease_weights = {d: custom_weights.get(d, default_weight) for d in valid_diseases}
total_weight = sum(disease_weights.values())
disease_probs = [disease_weights[d] / total_weight for d in valid_diseases]

# 10. Core & Optional Symptoms (universal always in optional!)
core_frac = 0.7  # 70% core, 30% optional (except universal)
core_symptoms = {}
optional_symptoms = {}

for disease, symptoms in disease_symptom_map.items():
    symptoms_clean = list(set([s for s in symptoms if s and s != 'nan']))
    # Split core & optional (remove universal from core, always add them to optional)
    core_n = max(1, int(core_frac * len(symptoms_clean)))
    random.shuffle(symptoms_clean)
    core = set(symptoms_clean[:core_n]) - universal_symptoms
    optional = set(symptoms_clean[core_n:]) | (set(symptoms_clean[:core_n]) & universal_symptoms)
    core_symptoms[disease] = core
    optional_symptoms[disease] = optional

# 11. Realistic Age Distribution
age_buckets = [(1, 12), (13, 19), (20, 35), (36, 55), (56, 90)]
age_weights = [0.08, 0.10, 0.25, 0.30, 0.27]

# 12. Generate Patient Dataset
num_rows = 100000  # Change as needed
final_rows = []

print(f"🔁 Generating {num_rows:,} patient rows...")
for i in tqdm(range(num_rows)):
    disease = random.choices(valid_diseases, weights=disease_probs, k=1)[0]
    core = core_symptoms.get(disease, set())
    opt = optional_symptoms.get(disease, set())
    row = {}
    # Unique patient ID (optional but helpful!)
    row['patient_id'] = f"PT{i+1:06d}"

    # --- Age group logic ---
    age_range = random.choices(age_buckets, weights=age_weights, k=1)[0]
    row['age'] = random.randint(*age_range)
    if row['age'] <= 12:
        age_group = 'child'
    elif row['age'] >= 56:
        age_group = 'elderly'
    else:
        age_group = 'adult'
    row['age_group'] = age_group

    # --- Symptom cluster logic ---
    used_cluster = False
    # Use age-specific clusters if available for this disease
    if disease in age_specific_clusters and age_group in age_specific_clusters[disease] and random.random() < 0.7:
        cluster = random.choice(age_specific_clusters[disease][age_group])
        for s in cluster:
            if s in all_symptoms:
                row[s] = 1
        used_cluster = True
    elif disease in disease_clusters and random.random() < 0.7:
        cluster = random.choice(disease_clusters[disease])
        for s in cluster:
            if s in all_symptoms:
                row[s] = 1
        used_cluster = True

    for s in all_symptoms:
        if s in row:  # already set by cluster
            continue
        if s in core:
            row[s] = int(random.random() < random.uniform(0.9, 0.98))
        elif s in opt:
            row[s] = int(random.random() < random.uniform(0.18, 0.45))
        else:
            # Reduce chance of classic cold/flu cluster symptoms appearing together for unrelated diseases
            if s in {'fever', 'headache', 'runny_nose', 'sneezing', 'sore_throat', 'cough'}:
                row[s] = int(random.random() < 0.001)
            else:
                row[s] = int(random.random() < random.uniform(0.001, 0.005))

    row['gender'] = random.choice(['M', 'F'])
    row['disease'] = disease
    row['precaution'] = precaution_map.get(disease, "Consult a doctor")
    row['doctor_type'] = doctor_map.get(disease, "General Physician")
    row['total_symptoms'] = sum(row[s] for s in all_symptoms)

    final_rows.append(row)

# 13. Save to CSV (standard name)
final_df = pd.DataFrame(final_rows)
final_df.to_csv('outputs/p.csv', index=False)
print("✅ All done! Realistic dataset saved as patient_centric.csv")
