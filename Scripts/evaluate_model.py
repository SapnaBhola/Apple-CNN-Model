import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your trained model
model_path = "/content/drive/MyDrive/Apple_dataset/final_apple_model.keras"
model = tf.keras.models.load_model(model_path)

# Set path to test data
test_dir = "/content/drive/MyDrive/Apple_dataset/test"

# Image size and batch size
img_size = (256, 256)
batch_size = 32

# Create ImageDataGenerator for test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Get predictions
y_pred = model.predict(test_generator, verbose=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# True labels
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Print metrics
print("Accuracy:", accuracy_score(y_true, y_pred_labels))
print("Precision:", precision_score(y_true, y_pred_labels, average='weighted'))
print("Recall:", recall_score(y_true, y_pred_labels, average='weighted'))
print("F1 Score:", f1_score(y_true, y_pred_labels, average='weighted'))

# Classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred_labels, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
