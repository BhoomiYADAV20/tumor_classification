import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import time
import base64
import os

# Load model
model = load_model("resnet50_best_model.h5")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Page configuration
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")

# Custom Header
st.markdown("<h1 style='text-align: center; color: #5eaaa8;'>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered ResNet50 model to classify tumor types from MRI scans</p>", unsafe_allow_html=True)

# Sidebar: Project Overview
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    - Classifies MRI scans into:
      - 🧠 Glioma
      - 💀 Meningioma
      - 🧪 Pituitary
      - ✅ No Tumor
    - Model: ResNet50 + Dense Head
    - Built with: TensorFlow & Streamlit  
    - Developer: Bhoomi Yadav
    """)
    st.markdown("---")
    st.markdown("📌 Sample Labels")
    for label in class_names:
        st.markdown(f"• {label}")

# Upload Section
st.subheader("📤 Upload an MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded MRI Image", use_column_width=True)

    # Prediction button
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing MRI image..."):
            time.sleep(1)  # simulate loading

            # Preprocess
            img_resized = image.resize((224, 224))
            img_array = img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred)

        # Display result
        st.success(f"🎯 Predicted Tumor Type: {predicted_class}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # Horizontal bar chart
        st.subheader("📈 Prediction Probability")
        prob_chart = {class_names[i]: float(pred[i]) for i in range(len(class_names))}
        fig, ax = plt.subplots(figsize=(6, 2))
        bars = ax.barh(list(prob_chart.keys()), list(prob_chart.values()), color="#5eaaa8")
        ax.set_xlim(0, 1)
        ax.bar_label(bars, fmt="%.2f")
        st.pyplot(fig)

# Expanders
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    - Pretrained ResNet50 extracts features from the MRI scan.
    - Fully-connected layers classify the image into one of the 4 categories.
    - This model was trained on the 4-Class Brain Tumor MRI dataset (Kaggle).

    Model input: 224x224 RGB MRI Image  
    Model output: 4-class softmax probabilities
    """)

with st.expander("🧠 Model Summary"):
    st.markdown("""
    | Layer            | Type         | Output Shape      |
    |------------------|--------------|-------------------|
    | Base Model       | ResNet50     | 7x7x2048          |
    | Global Pooling   | GAP          | 2048              |
    | Dense (128)      | ReLU         | 128               |
    | Dropout (0.3)    | Dropout      | 128               |
    | Output (4)       | Softmax      | 4 (class probs)   |
    """)

# Example Image Carousel (if you have sample images locally)
if os.path.exists("samples"):
    st.subheader("🧪 Example MRI Samples")
    sample_images = os.listdir("samples")
    cols = st.columns(4)
    for i, img_file in enumerate(sample_images[:4]):
        with cols[i]:
            img = Image.open(os.path.join("samples", img_file))
            st.image(img, caption=img_file.split('.')[0], use_column_width=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2025 | Built with ❤️ by Bhoomi Yadav | Powered by Streamlit</p>", unsafe_allow_html=True)
