# Streamlit Dual-App Suite: Face Detection Mobile App & ML Assignment 2

## Overview
A unified, Streamlit Cloud-ready web and mobile-responsive application housing two distinct applications selectable from a central launchpad dashboard:
1. **Instance 1: Face Detection Mobile Application** — Mobile-friendly computer vision interface supporting phone camera snapshots (`st.camera_input`), image uploads, and sample portraits, with configurable OpenCV detection parameters (faces, eyes, smiles), cropped face extraction, and downloadable annotated outputs.
2. **Instance 2: ML Assignment 2 (Build & Analyse an ML Model)** — End-to-end interactive Machine Learning workflow covering Dataset Selection, Problem Statement & EDA, Data Preprocessing, Model Selection & Training, Evaluation Metrics, Visual Analytics (Confusion Matrix, ROC Curve, Feature Importance, Distribution Plots), and Result Analysis.

## Project Type
**WEB / DATA SCIENCE (Streamlit Multi-App Dashboard — Mobile-Responsive)**
- **Primary Agents:** `project-planner`, `orchestrator`, `frontend-specialist`, `backend-specialist`
- **Key Skills:** `clean-code`, `app-builder`, `plan-writing`, `brainstorming`, `python-patterns`

## Success Criteria
- [x] Unified landing dashboard allowing instant switching between **Face Detection Mobile App** and **ML Assignment 2**.
- [x] **App 1 (Face Detection)** works seamlessly on mobile & desktop browsers using `st.camera_input` and file upload without `libGL` errors on Streamlit Cloud.
- [x] **App 2 (ML Assignment 2)** fulfills all 8 assignment rubric points:
  1. Suitable dataset selection
  2. Clear problem statement & dataset explanation
  3. Data preprocessing demonstration (missing values, scaling, train/test split)
  4. ML algorithm selection & hyperparameter tuning
  5. Model training & testing execution
  6. Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
  7. Interactive graphs/plots (Confusion Matrix, ROC Curve, Feature Importance, Correlation Heatmap)
  8. Structured analytical interpretation of the results
- [x] Easily deployable on **Streamlit Community Cloud (Free Tier)** via GitHub (`requirements.txt` with `opencv-python-headless`, `packages.txt` if needed, `.gitignore`, `README.md`).

## Tech Stack
| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Framework** | `streamlit` | Unified interactive dashboard & native Streamlit Cloud deployment |
| **Computer Vision** | `opencv-python-headless`, `Pillow`, `numpy` | Headless OpenCV avoids `libGL.so.1` crashes on Streamlit Cloud Linux containers |
| **Machine Learning** | `scikit-learn`, `pandas`, `numpy` | Industry-standard preprocessing, model training, cross-validation, and metrics |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` | High-clarity charts for EDA, Confusion Matrix, ROC curves, and Feature Importance |
| **Deployment** | Git + GitHub + Streamlit Community Cloud | Zero-cost cloud hosting directly from GitHub repository |

## File Structure
```text
ML_Assignment/
├── app.py                          # Main entrypoint & Landing Dashboard selector
├── modules/
│   ├── __init__.py
│   ├── face_detection.py           # Instance 1: Mobile-friendly Face Detection App
│   └── ml_assignment.py            # Instance 2: End-to-End ML Model Build & Analysis
├── utils/
│   ├── __init__.py
│   └── sample_generator.py         # Synthetic/sample face illustration & dataset helpers
├── .streamlit/
│   └── config.toml                 # Theme & mobile-friendly server configuration
├── requirements.txt                # Pinned Streamlit Cloud compatible dependencies
├── packages.txt                    # OS-level packages for Streamlit Cloud (if needed)
├── .gitignore                      # Python & OS ignore rules
├── README.md                       # Setup, architecture, and Streamlit Cloud deployment guide
└── streamlit-ml-suite.md           # Project Plan (this file)
```

## Task Breakdown

### Task 1: Project Configuration & Streamlit Cloud Setup
- **Agent:** `devops-engineer` | **Skill:** `python-patterns`
- **Priority:** P0 | **Dependencies:** None
- **INPUT:** Empty workspace
- **OUTPUT:** `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, `.gitignore`
- **VERIFY:** `pip install -r requirements.txt` succeeds without dependency conflicts.

### Task 2: Central Dashboard & Navigation Shell (`app.py`)
- **Agent:** `frontend-specialist` | **Skill:** `clean-code`
- **Priority:** P0 | **Dependencies:** Task 1
- **INPUT:** User navigation requirements
- **OUTPUT:** `app.py` with interactive Launchpad cards + sidebar navigation to switch between Home, App 1 (Face Detection), and App 2 (ML Assignment 2).
- **VERIFY:** Launching `streamlit run app.py` loads the chooser interface cleanly.

### Task 3: Instance 1 — Face Detection Mobile Application (`modules/face_detection.py`)
- **Agent:** `backend-specialist` + `frontend-specialist` | **Skill:** `python-patterns`, `clean-code`
- **Priority:** P1 | **Dependencies:** Task 2
- **INPUT:** Camera snapshot (`st.camera_input`), uploaded image (`st.file_uploader`), or built-in sample image.
- **OUTPUT:** `modules/face_detection.py` featuring OpenCV Haar Cascade detection (Faces, Eyes, Smiles), mobile-first UI controls (`scaleFactor`, `minNeighbors`, bounding box styling), extracted face gallery, detection metrics, and image download button.
- **VERIFY:** Test face detection pipeline via script on a test image and verify bounding boxes and face counts.

### Task 4: Instance 2 — ML Assignment 2 Pipeline (`modules/ml_assignment.py`)
- **Agent:** `backend-specialist` | **Skill:** `python-patterns`, `clean-code`
- **Priority:** P1 | **Dependencies:** Task 2
- **INPUT:** Dataset selection & ML hyperparameters
- **OUTPUT:** `modules/ml_assignment.py` structured into clear sections covering all 8 rubric requirements:
  1. Dataset Selection & Overview
  2. Problem Statement & Feature Dictionary
  3. Data Preprocessing (Imputation, Scaling, Train/Test Split inspection)
  4. Algorithm Selection (Random Forest, Logistic Regression, SVM, Gradient Boosting, KNN)
  5. Model Training & Testing with timing/status metrics
  6. Evaluation Metrics Dashboard (Accuracy, Precision, Recall, F1, ROC-AUC, Classification Report)
  7. Visual Plots (Target Distribution, Correlation Heatmap, Confusion Matrix, ROC Curve, Feature Importance)
  8. Automated & Written Analytical Summary of Results
- **VERIFY:** Execute ML pipeline function across supported models and confirm all metrics and figures render without errors.

### Task 5: Git Initialization, Verification & GitHub Push Readiness
- **Agent:** `test-engineer` + `devops-engineer` | **Skill:** `verify-changes`
- **Priority:** P2 | **Dependencies:** Tasks 3, 4
- **INPUT:** Completed codebase
- **OUTPUT:** Unit/smoke tests, `README.md`, initialized Git repo committed and pushed (or ready for remote URL).
- **VERIFY:** Run automated smoke test + Streamlit headless boot check.

## Phase X: Final Verification Checklist
- [x] Security Scan (`security_scan.py`)
- [x] Lint & Syntax Check (`py_compile` / `lint_runner.py`)
- [x] Smoke Test of Face Detection & ML Pipeline modules
- [x] Streamlit Server Runtime Verification
- [x] Git Repository & Streamlit Cloud Compatibility Check

## ✅ PHASE X COMPLETE
- Lint: ✅ Pass (`ruff` — 0 errors)
- Security: ✅ No critical issues (`[OK] SECURE`)
- Build & Unit Tests: ✅ Success (`2/2` tests passed)
- Date: 2026-10-28
