# Machine Learning Assignment

A Streamlit project containing two applications selectable from the main page:

1. **Face Detection Application (`modules/face_detection.py`)**
   - Uses OpenCV Haar Cascades (`opencv-python-headless`) to detect faces, eyes, and smiles.
   - Supports image upload, camera input (`st.camera_input`), and sample portraits.

2. **ML Assignment 2: Build and Analyse an ML Model (`modules/ml_assignment.py`)**
   - **1. Dataset Selection:** Breast Cancer Wisconsin (Diagnostic) and UCI Wine datasets.
   - **2. Problem Statement & Dataset Explanation:** Overview, dimensions, class distribution, and summary statistics.
   - **3. Data Preprocessing:** Median missing-value imputation, stratified train-test split, and feature scaling (`StandardScaler` / `MinMaxScaler`).
   - **4. ML Algorithm Selection:** Random Forest, Logistic Regression, SVM, Gradient Boosting, and KNN.
   - **5. Model Training and Testing:** Fits on the training set and evaluates on the unseen test set.
   - **6. Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Classification Report.
   - **7. Graphs and Plots:** Confusion Matrix, ROC Curve, Feature Importance chart, and Correlation Heatmap.
   - **8. Short Analysis of Results:** Summary of accuracy, error counts, key features, and generalization performance.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/), select your repository, and set the main file path to `app.py`.
