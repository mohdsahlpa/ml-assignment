# ⚡ ML & Computer Vision Dual-Instance Streamlit Suite

A unified, mobile-responsive **Streamlit** project containing two standalone application interfaces accessible from a central **Launchpad Dashboard**. Designed for effortless one-click deployment on **Streamlit Community Cloud (Free Tier)** directly from GitHub.

---

## 🚀 Applications Included

### 1. 📱 Instance 01: Face Detection Mobile Application (`modules/face_detection.py`)
A mobile-first computer vision interface powered by **OpenCV Haar Feature-based Cascade Classifiers** (`opencv-python-headless` for full Streamlit Cloud Linux compatibility):
- **Mobile Camera Snapshot (`st.camera_input`):** Capture live photos directly from a smartphone front/rear camera or laptop webcam via the browser.
- **Image Upload (`st.file_uploader`):** Upload JPG, PNG, or WEBP portraits from your mobile gallery or desktop.
- **Built-in Calibrated Sample Portraits:** Test single and dual-person portrait scenes out of the box without needing a camera.
- **Multi-Cascade Biometric Detection:** Simultaneously detects **Frontal Faces**, **Ocular Regions (Eyes)**, and **Smile Signatures**.
- **Interactive Parameter Tuning & Privacy Controls:** Adjust `scaleFactor`, `minNeighbors`, `minSize`, and apply optional **Gaussian Blur** or **Pixelate Mosaic** face anonymization.
- **Extracted Face ROI Gallery & PNG Download:** Inspect individual cropped faces with bounding-box telemetry and download the annotated high-contrast HUD frame.

---

### 2. 📊 Instance 02: ML Assignment 2 — Build & Analyse an ML Model (`modules/ml_assignment.py`)
A structured, interactive Machine Learning laboratory covering all 8 assignment rubric requirements:

1. **Select a Suitable Dataset:**
   - Primary: **Breast Cancer Wisconsin (Diagnostic) Dataset** (`569` biopsy instances, `30` cytological features, binary target: `Malignant (0)` vs. `Benign (1)`).
   - Secondary selectable benchmark: **UCI Wine Chemical Recognition Dataset** (`178` instances, `13` physicochemical features, `3` classes).
2. **Explain the Problem Statement & Dataset:**
   - Dedicated clinical problem formulation, class distribution breakdown, interactive raw data preview (`df.head()`), and statistical summary (`df.describe()`).
3. **Perform Data Preprocessing:**
   - Median missing-value imputation (`SimpleImputer`) with interactive synthetic missing-data simulation (`0%–10%`) to verify pipeline resilience.
   - Configurable feature standardization (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) with before-and-after mean/std verification tables.
   - Stratified train/test splitting (`15%–40%` configurable test split).
4. **Select an Appropriate ML Algorithm:**
   - Interactive choice between **Random Forest Classifier**, **Logistic Regression**, **Support Vector Machine (SVM)**, **Gradient Boosting Classifier**, and **K-Nearest Neighbors (KNN)** with hyperparameter controls and architectural rationale.
5. **Train and Test the Model:**
   - Fits the selected estimator on the scaled training split, evaluates on the held-out test split, and runs **5-Fold Stratified Cross-Validation**.
6. **Calculate Suitable Evaluation Metrics:**
   - Displays **Test Accuracy**, **Precision**, **Recall (Sensitivity)**, **F1-Score**, **ROC-AUC Score**, **5-Fold CV Mean ± Std**, per-class **Classification Report**, and a **Multi-Algorithm Benchmark Table** comparing all 5 models side-by-side.
7. **Display Results Using Graphs / Plots:**
   - **Confusion Matrix Heatmap**
   - **Receiver Operating Characteristic (ROC) Curve & AUC**
   - **Top 12 Predictive Feature Importance / Coefficient Bar Chart**
   - **Feature Correlation Matrix Heatmap**
8. **Short Analysis of the Results:**
   - Executive analytical interpretation synthesizing generalization gap (train vs. test accuracy), confusion matrix error breakdown (False Positives vs. False Negatives), top cytological biomarkers, and preprocessing impact.

---

## 📂 Project Structure

```text
ML_Assignment/
├── app.py                          # Main entrypoint & Landing Dashboard selector
├── modules/
│   ├── __init__.py
│   ├── face_detection.py           # Instance 1: Face Detection Mobile App
│   └── ml_assignment.py            # Instance 2: ML Assignment 2 (Build & Analyse Model)
├── utils/
│   ├── __init__.py
│   └── sample_generator.py         # Calibrated synthetic portrait generator
├── tests/
│   └── test_suite.py               # Automated unit & integration tests
├── .streamlit/
│   └── config.toml                 # Theme & server configuration
├── requirements.txt                # Streamlit Cloud Python dependencies
├── packages.txt                    # Streamlit Cloud Debian/Ubuntu apt packages
└── README.md                       # Documentation & deployment guide
```

---

## 💻 Local Installation & Execution

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```

---

## ☁️ Deploying on Streamlit Community Cloud (Free Tier via GitHub)

1. Push this repository to your **GitHub** account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
3. Click **Create app** → Select your repository, branch (`main`), and set **Main file path** to:
   ```text
   app.py
   ```
4. Click **Deploy!** Streamlit Cloud will automatically read `requirements.txt` (`opencv-python-headless`, `scikit-learn`, etc.) and `packages.txt` (`libgl1`, `libglib2.0-0`) and host your live application with a shareable URL that works on both mobile phones and desktop browsers.
