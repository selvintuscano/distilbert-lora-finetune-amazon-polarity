import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# === Load Model and Tokenizer ===
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
    return model, tokenizer

model, tokenizer = load_model()
id2label = {0: "Negative", 1: "Positive"}

# === Streamlit UI ===
st.title("🧠 Amazon Review Sentiment Classifier")
st.markdown("This app predicts whether a product review is **Positive** or **Negative**.")

user_input = st.text_area("✍️ Enter your review:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        label = id2label[prediction]

        st.markdown(f"### 🎯 Prediction: **{label}**")
