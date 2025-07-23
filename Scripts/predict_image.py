import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained Apple disease model
model_path = "/content/drive/MyDrive/Apple_dataset/final_apple_model.keras"
model = load_model(model_path)

# Provide the full path to your test image
test_image_path = "/content/drive/MyDrive/Apple_dataset/test/black_rot/02168189-aa75-4284-a7f0-8ca5901ea783___JR_FrgE.S 2948_90deg.JPG"

# ✅ Define class labels used during training
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy'
]

# 🔍 Known actual class of this test image (manually specify or read from filename)
actual_class_name = 'Apple___Black_rot'  # ← Update this as per your test image

# Read and preprocess the image
image = cv2.imread(test_image_path)
if image is None:
    print("❌ Error: Image not found at the specified path.")
else:
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    predicted_class_name = class_labels[predicted_label]

    # ✅ Print both actual and predicted class
    print(f"✅ Actual Class:    {actual_class_name}")
    print(f"✅ Predicted Class: {predicted_class_name}")
