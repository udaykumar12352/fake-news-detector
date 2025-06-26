import streamlit as st
from app.web_scraper import extract_text_from_url
from app.predictor import predict

st.title("📰 Fake News Detector")

choice = st.radio("Input Method", ["Paste Text", "Paste URL"])
if choice == "Paste Text":
    content = st.text_area("Enter news content")
else:
    url = st.text_input("Enter article URL")
    if st.button("Extract"):
        content = extract_text_from_url(url)
        st.text_area("Extracted Article", value=content, height=200)

if st.button("Predict"):
    label, probs = predict(content)
    st.write("Prediction:", "Real" if label else "Fake")
    st.write("Confidence:", f"{probs[label]*100:.2f}%")
