import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array
from PIL import Image
import base64
from io import BytesIO

# Load the trained model
model_path = "/content/drive/MyDrive/Apple_dataset/final_apple_model.keras"
model = tf.keras.models.load_model(model_path)

class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy'
]

care_recommendations = {
    'Apple___Apple_scab': [
        "Prune infected leaves and twigs.",
        "Apply fungicide in early spring.",
        "Avoid overhead irrigation."
    ],
    'Apple___Black_rot': [
        "Remove and destroy infected fruit and leaves.",
        "Use fungicide sprays during the growing season.",
        "Clean up plant debris regularly."
    ],
    'Apple___Cedar_apple_rust': [
        "Remove nearby cedar trees if possible.",
        "Use resistant apple varieties.",
        "Apply fungicides during early spring."
    ],
    'Apple___healthy': [
        "Maintain proper spacing for airflow.",
        "Regularly inspect leaves for early signs of disease.",
        "Ensure proper fertilization and watering schedule."
    ]
}

def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.set_page_config(page_title="Apple Leaf Disease Detector", page_icon="🌿", layout="centered")

st.markdown("""<style>
.result-card {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
.result-card.healthy {
    background-color: #e0f7e9;
    border: 1px solid #34a853;
}
.result-card.diseased {
    background-color: #ffe0e0;
    border: 1px solid #ea4335;
}
.confidence-meter {
    background-color: #f0f0f0;
    border-radius: 20px;
    height: 20px;
    margin-top: 10px;
}
.confidence-fill {
    background-color: #34a853;
    height: 100%;
    border-radius: 20px;
}
</style>""", unsafe_allow_html=True)

st.markdown("""
    <div class="header">
        <h1><b>🌿 Botanical Care:</b></h1>
        <h2>A Deep Learning Approach for Apple Leaf Disease Detection</h2>
        <p>Upload a leaf image to check for common diseases</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an apple leaf image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    with st.spinner('Analyzing your leaf...'):
        # Load image and show original size using HTML
        pil_image = Image.open(uploaded_file)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f"""
            <div style="text-align:center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto;"/>
            </div>
        """, unsafe_allow_html=True)

        # Prepare for prediction
        image_np = np.array(pil_image.convert('RGB'))
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        processed_image = preprocess_image(image_cv2)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.markdown(f'<div class="result-card {"healthy" if "healthy" in predicted_class else "diseased"}">', unsafe_allow_html=True)

        threshold = 60
        if confidence < threshold:
            st.warning("Unable to confidently detect disease. Try uploading a clearer apple leaf image.")
        else:
            if "healthy" in predicted_class:
                st.success("🍃 The apple leaf is healthy.")
            else:
                st.error(f"🚨 Disease Detected: {predicted_class.replace('___', ' ').title()}")

            st.write(f"**Confidence:** {confidence:.1f}%")
            st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width:{confidence}%"></div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("📌 Care Recommendations"):
                for tip in care_recommendations[predicted_class]:
                    st.write(f"- {tip}")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <small>For accurate diagnosis, consult an agricultural expert.</small>
    </div>
""", unsafe_allow_html=True)
