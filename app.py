import streamlit as st
import joblib

# Load model components
@st.cache_resource
def load_components():
    model = joblib.load("final_referral_model_FIXED.pkl")
    vectorizer = joblib.load("final_tfidf_vectorizer.pkl")
    label_encoder = joblib.load("final_label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_components()

# App title
st.title("AI Medical Referral Assistant")
st.markdown("Predict the **department or referral** based on a medical report.")

# Text input
input_text = st.text_area("Enter the medical report/description here:", height=200)

if st.button("Predict Referral"):
    if not input_text.strip():
        st.warning("Please enter some text to get a prediction.")
    else:
        # Transform and predict
        text_vec = vectorizer.transform([input_text])
        prediction = model.predict(text_vec)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Output
        st.success(f"🔍 **Predicted Referral Department:** {predicted_label}")


