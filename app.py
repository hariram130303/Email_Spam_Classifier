import streamlit as st
from model import train_models, predict

st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📧 Spam Email Classifier")
st.markdown("Enter a message below to classify it as **Spam** or **Ham**.")

tfidf, nb_model, svm_model = train_models()

msg = st.text_area("Message", height=150, placeholder="Type or paste an email or SMS message...")

model_choice = st.selectbox("Choose Classifier", ("Naive Bayes", "SVM"))

if st.button("Classify"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        model = nb_model if model_choice == "Naive Bayes" else svm_model
        result = predict(msg, model, tfidf)
        st.success(f"Result: **{result}**")
