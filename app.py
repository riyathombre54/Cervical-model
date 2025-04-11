# app.py
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
model_path = "cervical_model"
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.eval()
id2label = model.config.id2label

st.set_page_config(page_title="Predict", layout="centered")
st.title("🔍 Predict Cervical Cell Status")
st.write("Upload a cervical cell image to classify it as *normal* or *abnormal*.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Predict"):
        with st.spinner("Analyzing..."):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_idx = outputs.logits.argmax(-1).item()
                prediction = id2label[str(predicted_idx)]

        st.success(f"🩺 *Prediction: {prediction.upper()}*")
