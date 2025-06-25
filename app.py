import streamlit as st
import joblib

# تحميل الملفات
model = joblib.load("final_referral_model_FIXED.pkl")
vectorizer = joblib.load("final_tfidf_vectorizer.pkl")
label_encoder = joblib.load("final_label_encoder.pkl")

st.title("AI Referral Decision Support System")
st.markdown("This system decides whether the patient should be referred **inside** or **outside** the hospital.")

# إدخال المستخدم
user_input = st.text_area("Enter the medical report:")

if st.button("Predict Referral Type"):
    if user_input.strip() == "":
        st.warning("Please enter a valid medical report.")
    else:
        # تحويل النص إلى أرقام
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)
        label = label_encoder.inverse_transform(prediction)[0]

        if label == "داخل":
            st.success("✅ Recommendation: Refer **inside** the hospital.")
        elif label == "خارج":
            st.warning("⚠️ Recommendation: Refer **outside** the hospital.")
        else:
            st.info(f"Recommendation: {label}")
