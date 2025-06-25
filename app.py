
import streamlit as st
import joblib
import numpy as np

# Load model and components
model = joblib.load("final_referral_model.pkl")
vectorizer = joblib.load("final_tfidf_vectorizer.pkl")
label_encoder = joblib.load("final_label_encoder.pkl")

# Risk keywords for alerts
risk_keywords = ["seizure", "mass", "unconscious", "bleeding", "paralysis", "loss of vision", "rapid deterioration"]

st.title("🧠 Smart Referral Decision Support System")
st.markdown("Analyze patient case reports and receive intelligent referral recommendations with alerts.")

# Inputs
referral_note = st.text_area("🔍 Enter Referral Note", height=150)
specialty = st.text_input("🏥 Medical Specialty Required")
hospital = st.text_input("🏨 Current Hospital")
repeat_visit = st.checkbox("🔁 Has the patient visited recently (past 30 days)?")

if st.button("Analyze Case"):
    if not referral_note or not specialty or not hospital:
        st.warning("Please complete all fields.")
    else:
        full_text = referral_note + " " + specialty + " " + hospital
        vectorized_text = vectorizer.transform([full_text])

        # Risk detection
        keyword_alert = any(kw in referral_note.lower() for kw in risk_keywords)
        alert_flag = int(keyword_alert)
        repeat_flag = int(repeat_visit)

        # Combine all features
        alerts = np.array([[alert_flag, repeat_flag]])
        final_input = np.hstack([vectorized_text.toarray(), alerts])

        # Predict
        prediction = model.predict(final_input)
        decision = label_encoder.inverse_transform(prediction)[0]

        st.subheader(f"✅ AI Decision: **{decision}**")

        if keyword_alert:
            st.error("⚠️ Alert: Critical symptoms detected in referral note.")
        if repeat_flag:
            st.warning("🔁 Notice: Patient has repeated hospital visits recently.")
