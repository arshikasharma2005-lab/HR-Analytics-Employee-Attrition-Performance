import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time

# Page config
st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="✍️",
    layout="centered"
)

# Custom CSS for colors
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #667eea, #764ba2);
}
.big-font {
    font-size:30px !important;
    color:#4CAF50;
    font-weight:bold;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#f9f9f9;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("digit_model.h5")

# Sidebar
st.sidebar.title("🧠 About Project")
st.sidebar.info(
    "This AI app uses a **Convolutional Neural Network (CNN)** "
    "trained on the **MNIST dataset** to recognize handwritten digits (0–9)."
)
st.sidebar.markdown("### 🛠 Tech Stack")
st.sidebar.write("- Python\n- TensorFlow\n- Streamlit\n- OpenCV")

# Main title
st.markdown('<p class="big-font">✍️ Handwritten Digit Recognition</p>', unsafe_allow_html=True)
st.write("Upload an image of a handwritten digit and let AI predict it 🎯")

# Upload section
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="🖼 Uploaded Image", width=200)

        # Preprocess
        img = np.array(image)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28)

        if st.button("🔍 Predict Digit"):
            with st.spinner("🤖 AI is thinking..."):
                time.sleep(1)

                prediction = model.predict(img)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100

            st.success(f"🎯 **Predicted Digit:** {digit}")
            st.info(f"📊 **Confidence:** {confidence:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# Reset
if st.button("🔄 Reset App"):
    st.experimental_rerun()
