

```markdown
# 🍎 Apple Leaf Disease Classification using CNN

A deep learning-based image classification project to detect and categorize apple leaf diseases using custom CNN architecture. Built with **TensorFlow**, **Keras**, and deployed through **Streamlit**, this project showcases end-to-end ML engineering skills from model design to deployment.

---

## 💡 Project Summary

This application classifies apple leaf images into four categories:

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

It is designed to assist farmers, agriculturists, and researchers in identifying leaf diseases early for better crop management.

---

## 👩‍💻 Developer Role & Skills Demonstrated

- **Lead Developer**: Entire pipeline built and managed by **Sapna**
- **Skills Showcased**:
  - CNN model development
  - Data preprocessing & augmentation
  - Streamlit-based web interface
  - Model evaluation & visualization
  - Modular scripting for maintainability
- **Teamwork**: Contributions from **Inderjeet Kaur** and **Gourve** in planning, review, and feedback

---

## 🧠 Tech Stack

| Category             | Libraries                              |
|----------------------|----------------------------------------|
| Deep Learning        | TensorFlow 2.18.0, Keras 3.8.0          |
| Image Processing     | OpenCV 4.12.0, Pillow 11.2.1            |
| Data Science         | NumPy 2.0.2, Pandas 2.2.2               |
| Visualization        | Matplotlib 3.10.0                       |
| Metrics              | Scikit-learn 1.6.1                      |
| Web Frontend         | Streamlit 1.47.0                        |
| Python Version       | Requires Python 3.11+                   |

---

## 📁 Repository Structure

> 🔗 Click folder names to browse contents directly on GitHub.

```

.
├── [assets/](./assets)                   # Evaluation plots, saved models, UI images
├── [data/](./data)                       # Training and test datasets
│   ├── train/
│   └── test/
├── [notebooks/](./notebooks)            # Jupyter notebooks for model training & testing
│   ├── Apple\_CNN\_Model.ipynb
│   ├── Apple\_Evaluation.ipynb
│   └── Apple\_Testing\_Frontend.ipynb
├── [scripts/](./scripts)                # Python scripts for modular execution
│   ├── train\_model.py
│   ├── evaluate\_model.py
│   ├── predict\_image.py
│   └── web\_app.py
├── [LICENSE](./LICENSE)                 # MIT License
├── [.gitignore](./.gitignore)           # Git ignore rules
├── [requirements.txt](./requirements.txt) # Dependencies list
└── [README.md](./README.md)             # Project overview

````

---

## 📦 Dataset

- Source: [Apple Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/ludehsar/apple-disease-dataset)
- The dataset contains labeled images of apple leaves across 4 classes.
- Place the `train` and `test` folders under the `data/` directory as shown above.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SapnaBhola/Apple-CNN-Model.git
cd apple-leaf-cnn
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (if not already trained)

```bash
python scripts/train_model.py
```

### 4️⃣ Launch the Web App

```bash
streamlit run scripts/web_app.py
```

Upload any apple leaf image to predict its health condition in real-time.

---

## 📊 Sample Outputs

Located in the [`assets/`](./assets) folder:

* 📈 Accuracy & Loss graphs (training vs validation)
* 📉 Confusion matrix
* ✅ Sample input-output predictions
* 🖼️ Web UI screenshots

---

## 🎯 Key Features

* ✔️ Image classification using custom CNN
* ✔️ Real-time prediction with Streamlit interface
* ✔️ Modular design for training, evaluation & inference
* ✔️ Clean visualization and reproducible results

---

## 🌱 Use Cases

* Smart agriculture and disease prediction
* Plant pathology research support
* ML education and academic demonstration

---

## 🧪 Future Improvements

* 💬 Grad-CAM/SHAP integration for interpretability
* ☁️ Cloud-hosted web app version (Streamlit Cloud)
* 🧹 Enhanced dataset cleaning and augmentation

---

## 🤝 Acknowledgments

This project was developed by **Sapna**, with support from **Inderjeet Kaur** and **Gourve**.
All development, ML architecture, evaluation, deployment, and documentation were carried out by Sapna as part of a **professional learning portfolio project**.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 📬 Contact

* 📧 Email: sapna.bhola86@gmail.com
* 💼 LinkedIn: https://www.linkedin.com/in/sapna-18785b287/

---

> ⭐ Star this repo if you find it helpful or insightful.

