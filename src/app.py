import joblib
import numpy as np
import pandas as pd
import time
from train_model import get_symptom_map, map_symptom, emergency_check, predict, CLUSTERS
import io
from fpdf import FPDF
import re
import difflib
import streamlit as st
from PIL import Image
from random import choice
import os

# Set Streamlit to centered card layout
st.set_page_config(layout="centered", page_title="MediPredict", page_icon="🩺")

# --- Custom CSS for glassmorphism card and new color palette ---
st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700&display=swap');
    html, body, .main, .block-container {
        background: linear-gradient(120deg, #e0e7ff 0%, #f8fafc 100%);
        font-family: 'Inter', Arial, sans-serif !important;
    }
    .glass-card {
        background: linear-gradient(120deg, rgba(224,231,255,0.82) 0%, rgba(248,250,252,0.82) 100%);
        border-radius: 1.3em;
        box-shadow: 0 8px 32px 0 rgba(80, 80, 180, 0.18), 0 2px 8px 0 #7f53ac22;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1.5px solid rgba(180, 180, 255, 0.13);
        padding: 0.7em 1.5em 1.5em 1.5em;
        margin: 2.5em auto 2em auto;
        max-width: 440px;
        min-width: 320px;
        text-align: center;
        position: relative;
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .glass-card:hover {
        box-shadow: 0 12px 40px 0 rgba(80, 80, 180, 0.22), 0 4px 16px 0 #7f53ac33;
        transform: translateY(-2px) scale(1.012);
    }
    .mp-stepper {
        position: absolute;
        left: 50%;
        top: -38px;
        transform: translateX(-50%);
        background: rgba(255,255,255,0.85);
        box-shadow: 0 4px 18px 0 #7f53ac33, 0 1.5px 6px 0 #647dee22;
        border-radius: 2em;
        padding: 0.5em 2.2em 0.5em 2.2em;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5em;
        z-index: 10;
        min-width: 320px;
    }
    .main-title {
        font-family: 'Inter', Arial, sans-serif;
        font-size: 2em;
        font-weight: 700;
        background: linear-gradient(90deg, #7f53ac 0%, #647dee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1em;
        margin-top: 2.2em;
    }
    .subtitle {
        font-size: 1.05em;
        color: #7f53ac;
        margin-bottom: 1.1em;
        margin-top: 0.1em;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7f53ac 0%, #647dee 100%);
        color: white;
        border-radius: 1.2em;
        font-size: 1.05em;
        font-weight: 600;
        padding: 0.6em 2em;
        margin-top: 0.7em;
        box-shadow: 0 2px 8px #7f53ac33;
        transition: 0.2s;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
        box-shadow: 0 4px 16px #647dee33;
    }
    .stTextInput>div>div>input {
        border-radius: 1em;
        border: 1.2px solid #b0b0b0;
        padding: 0.6em;
        font-size: 1.05em;
        background: rgba(255,255,255,0.93);
        color: #222 !important;
    }
    .stTextInput>label, label[data-testid="stWidgetLabel"] {
        color: #b9a6e3 !important;
        font-weight: 700;
        font-size: 1.09em;
        letter-spacing: 0.01em;
        margin-bottom: 0.3em;
    }
    .stMarkdown {
        font-size: 1.08em;
    }
    .result-card {
        background: #f8f6ff;
        border-radius: 1.1em;
        box-shadow: 0 2px 12px #b0b0ff33;
        padding: 1em 1.2em;
        margin-bottom: 1em;
        border-left: 7px solid #7f53ac;
        color: #4b3c7a;
        text-align: left;
    }
    .confidence-badge {
        display: inline-block;
        font-size: 0.98em;
        font-weight: 700;
        padding: 0.25em 0.9em;
        border-radius: 1em;
        margin-right: 0.6em;
        color: #fff;
        background: #647dee;
    }
    .chip {
        background: #647dee;
        color: #fff;
        border-radius: 1em;
        padding: 0.25em 0.9em;
        font-size: 0.98em;
        margin-right: 0.4em;
        margin-bottom: 0.2em;
        display: inline-block;
    }
    /* Progress bar stepper */
    .mp-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-family: 'Inter', Arial, sans-serif;
    }
    .mp-step-circle {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #e0e7ff;
        color: #7f53ac;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1em;
        border: 2px solid #e0e7ff;
        margin-bottom: 0.1em;
        transition: background 0.2s, color 0.2s, border 0.2s;
    }
    .mp-step-circle.active {
        background: linear-gradient(90deg, #7f53ac 0%, #647dee 100%);
        color: #fff;
        border: 2px solid #7f53ac;
    }
    .mp-step-circle.completed {
        background: #43e97b;
        color: #fff;
        border: 2px solid #43e97b;
    }
    .mp-step-label {
        font-size: 0.93em;
        color: #7f53ac;
        font-weight: 600;
        margin-top: 0.05em;
        letter-spacing: 0.5px;
    }
    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: linear-gradient(120deg, #23243a 0%, #23243a 100%);
        box-shadow: 2px 0 16px #7f53ac22;
        border-right: 1.5px solid #2d2e4a;
    }
    /* Green download button */
    .mp-download-btn button {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #fff;
        border-radius: 1.2em;
        font-size: 1.08em;
        font-weight: 700;
        padding: 0.7em 2.2em;
        margin: 1.1em 0 1.2em 0;
        box-shadow: 0 2px 12px #43e97b33;
        border: none;
        transition: 0.2s;
    }
    .mp-download-btn button:hover {
        filter: brightness(1.08);
        box-shadow: 0 4px 18px #38f9d733;
    }
    </style>
''', unsafe_allow_html=True)

# --- Sidebar with logo, personalization, and quick tips ---
# Health tips/facts list
HEALTH_TIPS = [
    "Drink plenty of water every day.",
    "Wash your hands regularly to prevent illness.",
    "Get at least 7-8 hours of sleep each night.",
    "Regular exercise boosts your immune system.",
    "Eat a balanced diet rich in fruits and vegetables.",
    "Take breaks from screens to rest your eyes.",
    "Don't ignore persistent symptoms—consult a doctor.",
    "Practice deep breathing to reduce stress.",
    "Keep your vaccinations up to date.",
    "Early detection saves lives—don't delay checkups."
]

# User personalization
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'avatar_choice' not in st.session_state:
    st.session_state.avatar_choice = "Doctor"

with st.sidebar:
    # Doctor SVG logo (static, clear doctor look)
    st.markdown("""
    <div style='display:flex; justify-content:center; align-items:center; margin-bottom:0.7em;'>
      <img src='https://img.icons8.com/color/96/000000/doctor-male--v2.png' width='80' alt='Doctor'/>
    </div>
    """, unsafe_allow_html=True)
    # User name input and greeting
    if not st.session_state.user_name:
        user_name = st.text_input("Your name (optional)", "", max_chars=20, key="sidebar_name")
        if user_name:
            st.session_state.user_name = user_name
            st.rerun()
    else:
        st.markdown(f"<h3 style='color:#7f53ac; margin-bottom:0.2em;'>Hello, {st.session_state.user_name}!</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#647dee;font-size:1em;'>AI-powered Disease Prediction</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #e0e7ff;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.95em;'>Your privacy is our priority.<br>All predictions are confidential.</p>", unsafe_allow_html=True)
    # Quick tip/fact
    st.markdown("<div style='height:1.2em;'></div>", unsafe_allow_html=True)  # Spacer for visibility
    if 'last_tip' not in st.session_state or st.session_state.get('step', 1) != st.session_state.get('last_tip_step', 0):
        st.session_state.last_tip = choice(HEALTH_TIPS)
        st.session_state.last_tip_step = st.session_state.get('step', 1)
    st.markdown(f"""
    <div style='margin-top:1.2em; background:rgba(127,83,172,0.32); border-radius:1em; padding:1.25em 1.1em;'>
        <span style='color:#fff; font-weight:800; font-size:1.13em; text-shadow:0 1px 4px #7f53ac55;'>💡 Health Tip:</span><br>
        <span style='color:#fff; font-size:1.13em; font-weight:600; text-shadow:0 1px 4px #7f53ac55;'>{st.session_state.last_tip}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Main glass card with stepper at the very top ---
glass_card_html = '<div class="glass-card">'
# Stepper/Progress Bar
step_labels = ["Symptoms", "Related", "Severity", "Results"]
current_step = st.session_state.get('step', 1)
stepper_html = '<div class="mp-stepper">'
for i, label in enumerate(step_labels, 1):
    circle_class = "mp-step-circle"
    if i < current_step:
        circle_class += " completed"
    elif i == current_step:
        circle_class += " active"
    stepper_html += f'<div class="mp-step"><div class="{circle_class}">{i}</div><div class="mp-step-label">{label}</div></div>'
stepper_html += '</div>'
glass_card_html += stepper_html
# Main title and subtitle
main_title_html = '''
    <div class="main-title">🩺 MediPredict</div>
    <div class="subtitle">Describe your symptoms below to get a prediction and helpful advice.</div>
'''
glass_card_html += main_title_html
st.markdown(glass_card_html, unsafe_allow_html=True)

# --- Symptom autocomplete enhancement ---
ALL_SYMPTOMS = sorted(list({s for c in CLUSTERS for s in c['cluster']}))

# --- Initialize session state variables to prevent AttributeError ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = ""
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'suggested_symptoms' not in st.session_state:
    st.session_state.suggested_symptoms = []
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'severity' not in st.session_state:
    st.session_state.severity = 3

# --- Load precaution and doctor info ---
precaution_map = {}
doctor_map = {}
precaution_path = os.path.join('data', 'symptom_precaution.csv')
doctor_path = os.path.join('data', 'disease_centric_knowledgebase_with_doctor.csv')
if os.path.exists(precaution_path):
    df_prec = pd.read_csv(precaution_path)
    for _, row in df_prec.iterrows():
        disease = str(row.get('Disease', '')).strip().lower()
        precs = [str(row.get(f'Precaution_{i}', '')).strip() for i in range(1, 5) if row.get(f'Precaution_{i}', '')]
        precaution_map[disease] = [p for p in precs if p]
if os.path.exists(doctor_path):
    df_doc = pd.read_csv(doctor_path)
    # Try both 'Doctor' and 'doctor_type' columns for compatibility
    doc_col = 'Doctor' if 'Doctor' in df_doc.columns else ('doctor_type' if 'doctor_type' in df_doc.columns else None)
    if doc_col:
        for _, row in df_doc.iterrows():
            disease = str(row.get('Disease', '') or row.get('disease', '')).strip().lower()
            doc = str(row.get(doc_col, '')).strip()
            if disease and doc:
                doctor_map[disease] = doc

# --- Step 1: Symptom input ---
def go_to_next_step():
    # Merge selected and custom
    input_syms = [s.strip().lower().replace(' ', '_') for s in st.session_state.get('custom_input', '').split(',') if s.strip()]
    selected = st.session_state.get('selected_syms', [])
    all_syms = list(set(selected + input_syms))
    # Fuzzy match to features
    matched_syms = []
    for s in all_syms:
        match = difflib.get_close_matches(s, [f.lower() for f in ALL_SYMPTOMS], n=1, cutoff=0.7)
        if match:
            matched_syms.append(match[0])
        else:
            matched_syms.append(s)
    st.session_state.selected_symptoms = matched_syms
    # Find best cluster
    best_cluster = None
    best_count = 0
    for c in CLUSTERS:
        count = len([s for s in matched_syms if s in c['cluster']])
        if count > best_count:
            best_count = count
            best_cluster = c
    st.session_state.selected_cluster = best_cluster
    # Suggest missing symptoms from cluster
    if best_cluster:
        missing = [s for s in best_cluster['cluster'] if s not in matched_syms]
        st.session_state.suggested_symptoms = missing[:5]  # Suggest up to 5
    else:
        st.session_state.suggested_symptoms = []
    st.session_state.symptoms = st.session_state.get('custom_input', '')
    st.session_state.step = 2

if st.session_state.step == 1:
    st.markdown("""
    <span style='color:#4b3c7a; font-weight:600;'>Start typing to search and add symptoms. You can also enter custom symptoms separated by commas.</span>
    """, unsafe_allow_html=True)
    st.multiselect(
        "Select symptoms (autocomplete supported)",
        options=ALL_SYMPTOMS,
        default=st.session_state.selected_symptoms,
        help="Type to search, or enter custom symptoms below.",
        key="selected_syms"
    )
    st.text_input(
        "Or add custom symptoms (comma-separated)",
        st.session_state.symptoms,
        help="E.g. fever, cough, headache",
        key="custom_input",
        on_change=go_to_next_step
    )
    if st.button("Next ➡️", key="next_btn1"):
        go_to_next_step()

# --- Step 2: Suggest more symptoms from cluster ---
elif st.session_state.step == 2:
    # --- Cluster suggestion logic ---
    # On first entry to step 2, build a sorted list of clusters by match count
    if 'cluster_suggestion_idx' not in st.session_state:
        # Build cluster ranking
        user_syms = st.session_state.selected_symptoms
        cluster_scores = []
        for c in CLUSTERS:
            match_count = len([s for s in user_syms if s in c['cluster']])
            if match_count > 0:  # Only include clusters with at least one match
                cluster_scores.append((match_count, c))
        cluster_scores.sort(reverse=True, key=lambda x: x[0])
        st.session_state.cluster_suggestion_list = [c for score, c in cluster_scores]
        st.session_state.cluster_suggestion_idx = 0
    # Get current cluster to suggest from
    clusters = st.session_state.cluster_suggestion_list
    idx = st.session_state.cluster_suggestion_idx
    if clusters and idx < len(clusters):
        current_cluster = clusters[idx]
        missing = [s for s in current_cluster['cluster'] if s not in st.session_state.selected_symptoms]
        st.session_state.suggested_symptoms = missing[:5]
        cluster_name = current_cluster.get('disease', 'Cluster')
        st.markdown(f"""
        <span style='color:#7f53ac; font-weight:700; font-size:1.13em; text-shadow:0 1px 6px #e0e7ff99;'>Would you like to add any of these related symptoms from <b>{cluster_name}</b>?</span>
        """, unsafe_allow_html=True)
        added = False
        for i, s in enumerate(st.session_state.suggested_symptoms):
            label = s.replace('_', ' ').capitalize()
            btn_key = f"add_{idx}_{i}_{s}"
            if st.button(f"Add: {label}", key=btn_key):
                st.session_state.selected_symptoms.append(s)
                added = True
        if added:
            # Remove added from suggestions
            st.session_state.suggested_symptoms = [s for s in st.session_state.suggested_symptoms if s not in st.session_state.selected_symptoms]
            # Do NOT reset cluster suggestion to best match; stay on current cluster
            st.rerun()
        # Button to show next cluster's suggestions
        if st.button("No, none of these", key="next_cluster_btn"):
            st.session_state.cluster_suggestion_idx += 1
            st.rerun()
    else:
        st.markdown("<span style='color:#7f53ac; font-weight:700;'>No related suggestions found for your symptoms.</span>", unsafe_allow_html=True)
    st.markdown("""
    <span style='color:#7f53ac; font-weight:600;'>Click Next to specify severity for each symptom.</span>
    """, unsafe_allow_html=True)
    if st.button("Next ➡️", key="next_btn2"):
        st.session_state.step = 3
        # Clean up cluster suggestion state for next run
        if 'cluster_suggestion_idx' in st.session_state:
            del st.session_state['cluster_suggestion_idx']
        if 'cluster_suggestion_list' in st.session_state:
            del st.session_state['cluster_suggestion_list']
    if st.button("⬅️ Back", key="back_btn1"):
        st.session_state.step = 1
        # Clean up cluster suggestion state for next run
        if 'cluster_suggestion_idx' in st.session_state:
            del st.session_state['cluster_suggestion_idx']
        if 'cluster_suggestion_list' in st.session_state:
            del st.session_state['cluster_suggestion_list']

# --- Step 3: Per-symptom severity input ---
elif st.session_state.step == 3:
    st.markdown("""
    <span style='color:#7f53ac; font-weight:700; font-size:1.13em;'>Specify the severity for each symptom (1 = mild, 5 = severe):</span>
    """, unsafe_allow_html=True)
    if 'symptom_severity' not in st.session_state:
        st.session_state.symptom_severity = {s: 3 for s in st.session_state.selected_symptoms}
    # Update dict for new/removed symptoms
    for s in st.session_state.selected_symptoms:
        if s not in st.session_state.symptom_severity:
            st.session_state.symptom_severity[s] = 3
    for s in list(st.session_state.symptom_severity.keys()):
        if s not in st.session_state.selected_symptoms:
            del st.session_state.symptom_severity[s]
    valid = True
    for s in st.session_state.selected_symptoms:
        label = s.replace('_', ' ').capitalize()
        val = st.number_input(f"{label} severity", min_value=1, max_value=5, value=st.session_state.symptom_severity[s], step=1, key=f"sev_{s}")
        st.session_state.symptom_severity[s] = val
        if not (1 <= val <= 5):
            valid = False
    if st.button("Next ➡️", key="next_btn3"):
        if valid:
            st.session_state.step = 4
        else:
            st.warning("Please enter a severity between 1 and 5 for each symptom.")
    if st.button("⬅️ Back", key="back_btn2"):
        st.session_state.step = 2

# --- Step 4: Show prediction/results ---
elif st.session_state.step == 4:
    # Use selected symptoms and their severities to predict top 3 diseases
    user_symptoms = st.session_state.selected_symptoms
    user_severity_dict = st.session_state.symptom_severity
    # Use cluster logic to get top 3 diseases (optionally, you can use severity info here)
    cluster_scores = []
    for c in CLUSTERS:
        match_syms = [s for s in user_symptoms if s in c['cluster']]
        match_count = len(match_syms)
        total_cluster = len(c['cluster'])
        # Calculate average severity for matched symptoms
        if match_syms:
            avg_severity = np.mean([user_severity_dict.get(s, 3) for s in match_syms])
        else:
            avg_severity = 1
        # Confidence: weighted by match ratio and severity
        match_ratio = match_count / total_cluster if total_cluster else 0
        confidence = min(0.55 + 0.25*match_ratio + 0.12*(avg_severity/5), 0.99)
        cluster_scores.append((confidence, match_count, c['disease'], c))
    # Sort by confidence, then by match_count
    cluster_scores.sort(reverse=True)
    top3 = cluster_scores[:3]
    for confidence, score, disease, c in top3:
        # Normalize disease key for lookup
        disease_key = re.sub(r'\s+', ' ', disease.strip().lower())
        # Try alternate keys if not found
        precautions = precaution_map.get(disease_key, [])
        doctor = doctor_map.get(disease_key, None)
        # Defensive: try to match with/without underscores, spaces, etc.
        if not precautions:
            alt_key = disease_key.replace('_', ' ')
            precautions = precaution_map.get(alt_key, [])
        if not doctor:
            alt_key = disease_key.replace('_', ' ')
            doctor = doctor_map.get(alt_key, None)
        doctor_display = str(doctor).strip() if doctor else ''
        # Format precautions as point-wise list if available
        precaution_html = ''
        filtered_precs = [p for p in precautions if p and str(p).lower() != 'nan']
        if filtered_precs:
            precaution_html = '<div style="margin:0.5em 0 0.2em 0;"><b>Precautions:</b><ul style="margin:0.2em 0 0.2em 1.2em; padding-left:1.2em;">' + ''.join([f'<li style="text-align:left; color:#4b3c7a; font-size:1em;">{p}</li>' for p in filtered_precs]) + '</ul></div>'
        else:
            precaution_html = '<div style="margin:0.5em 0 0.2em 0;"><b>Precautions:</b> <span style="color:#888;">No specific precautions found.</span></div>'
        # Give advice based on presence of precautions
        if filtered_precs:
            advice = "Follow the listed precautions and consult a doctor if symptoms persist."
        else:
            advice = "Consult a doctor if symptoms persist."
        chips_html = ' '.join([
            f'<span class="chip">{s.replace("_", " ")}</span>'
            for s in user_symptoms
        ])
        # Doctor display logic: always show if mapped, with special note for General Physician
        doctor_html = ''
        if doctor_display and doctor_display.lower() != 'nan':
            if doctor_display.lower() == 'general physician':
                doctor_html = '<div style="margin:0.5em 0 0.2em 0;"><b>Doctor to visit:</b> General Physician <span style="color:#888;font-size:0.97em;">(You may consult any general physician.)</span></div>'
            else:
                doctor_html = f'<div style="margin:0.5em 0 0.2em 0;"><b>Doctor to visit:</b> {doctor_display}</div>'
        else:
            doctor_html = '<div style="margin:0.5em 0 0.2em 0;"><b>Doctor to visit:</b> <span style="color:#888;">Visit a nearby general physician.</span></div>'
        st.markdown(f"""
        <div class="result-card">
            <span class="confidence-badge">{int(confidence*100)}%</span>
            <span style="font-size:1.2em; font-weight:700;">🦠 {disease}</span>
            <div style="margin:0.7em 0 0.3em 0;"><b>Your symptoms:</b> {chips_html}</div>
            <div style="color:#4b3c7a;"><b>Advice:</b> {advice}</div>
            {precaution_html}
            {doctor_html}
        </div>
        """, unsafe_allow_html=True)
    # --- Custom CSS for green download button ---
    st.markdown('''
        <style>
        .mp-download-btn button {
            background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
            color: #fff;
            border-radius: 1.2em;
            font-size: 1.08em;
            font-weight: 700;
            padding: 0.7em 2.2em;
            margin: 1.1em 0 1.2em 0;
            box-shadow: 0 2px 12px #43e97b33;
            border: none;
            transition: 0.2s;
        }
        .mp-download-btn button:hover {
            filter: brightness(1.08);
            box-shadow: 0 4px 18px #38f9d733;
        }
        </style>
    ''', unsafe_allow_html=True)
    # --- Download Results Button ---
    if top3:
        # Prepare PDF content
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "MediPredict Results", ln=True, align='C')
        pdf.set_font("Arial", '', 12)
        for confidence, score, disease, c in top3:
            disease_key = re.sub(r'\s+', ' ', disease.strip().lower())
            precautions = precaution_map.get(disease_key, [])
            doctor = doctor_map.get(disease_key, None)
            if not precautions:
                alt_key = disease_key.replace('_', ' ')
                precautions = precaution_map.get(alt_key, [])
            if not doctor:
                alt_key = disease_key.replace('_', ' ')
                doctor = doctor_map.get(alt_key, None)
            doctor_display = str(doctor).strip() if doctor else ''
            filtered_precs = [p for p in precautions if p and str(p).lower() != 'nan']
            pdf.ln(4)
            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 10, f"{disease.title()} ({int(confidence*100)}%)", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Symptoms: {', '.join([s.replace('_', ' ') for s in user_symptoms])}", ln=True)
            advice = "Follow the listed precautions and consult a doctor if symptoms persist." if filtered_precs else "Consult a doctor if symptoms persist."
            pdf.multi_cell(0, 8, f"Advice: {advice}")
            if filtered_precs:
                pdf.cell(0, 8, "Precautions:", ln=True)
                for p in filtered_precs:
                    pdf.cell(0, 8, f"- {p}", ln=True)
            else:
                pdf.cell(0, 8, "Precautions: No specific precautions found.", ln=True)
            if doctor_display and doctor_display.lower() != 'nan':
                if doctor_display.lower() == 'general physician':
                    pdf.cell(0, 8, "Doctor to visit: General Physician (You may consult any general physician.)", ln=True)
                else:
                    pdf.cell(0, 8, f"Doctor to visit: {doctor_display}", ln=True)
            else:
                pdf.cell(0, 8, "Doctor to visit: Visit a nearby general physician.", ln=True)
            pdf.ln(2)
        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.markdown('<div class="mp-download-btn">', unsafe_allow_html=True)
        st.download_button(
            label="Download Results as PDF",
            data=pdf_output,
            file_name="medipredict_results.pdf",
            mime="application/pdf"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("⬅️ Back", key="back_btn3"):
        st.session_state.step = 3
    if st.button("🔄 Restart", key="restart_btn2"):
        st.session_state.step = 1
        st.session_state.symptoms = ""
        st.session_state.selected_symptoms = []
        st.session_state.suggested_symptoms = []
        st.session_state.selected_cluster = None
        st.session_state.severity = 3
        st.session_state.symptom_severity = {}

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;font-size:0.95em;color:#b0b0b0;'>
    <span>Powered by <b>MediPredict AI</b> | © 2025</span>
</div>
""", unsafe_allow_html=True)