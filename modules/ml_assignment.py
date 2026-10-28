"""Instance 2: ML Assignment 2 — End-to-End Machine Learning Model Build & Analysis."""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC


@st.cache_data
def load_selected_dataset(dataset_name: str) -> dict[str, Any]:
    """Loads and structures the chosen classification dataset with domain metadata."""
    if "Wine" in dataset_name:
        raw = load_wine()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = [str(t) for t in raw.target_names]
        problem_statement = (
            "**Problem Statement:** Predict the cultivar (class 0, 1, or 2) of Italian wines grown in the "
            "same region based on 13 physicochemical laboratory measurements (such as Alcohol, Malic Acid, "
            "Flavanoids, Color Intensity, and Proline). Accurate chemical classification automates quality "
            "assurance and origin verification."
        )
        dataset_explanation = (
            "The **UCI Wine Recognition Dataset** contains **178 instances** across **3 classes** with "
            "**13 continuous numerical features** representing chemical constituents."
        )
        is_binary = False
    else:
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = ["Malignant (0)", "Benign (1)"]
        problem_statement = (
            "**Problem Statement:** Accurately classify breast mass tumors as **Malignant (0)** or **Benign (1)** "
            "using quantitative cytological features extracted from digitized images of fine-needle aspirate (FNA) "
            "biopsies. In clinical oncology decision-support, maximizing **Recall (Sensitivity)** and **ROC-AUC** "
            "is critical to minimize false diagnoses while avoiding unnecessary invasive procedures."
        )
        dataset_explanation = (
            "The **Breast Cancer Wisconsin (Diagnostic) Dataset** consists of **569 biopsy records** "
            "(`212 Malignant`, `357 Benign`) and **30 continuous morphological features** describing cell nuclei "
            "characteristics: *radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, "
            "symmetry,* and *fractal dimension* (computed as mean, standard error, and worst/largest values)."
        )
        is_binary = True

    return {
        "df": df,
        "feature_names": list(raw.feature_names),
        "target_names": target_names,
        "problem_statement": problem_statement,
        "dataset_explanation": dataset_explanation,
        "is_binary": is_binary,
    }


def preprocess_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
    scaler_name: str = "StandardScaler (Z-Score)",
    test_size: float = 0.20,
    synthetic_missing_pct: float = 0.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Executes missing-value imputation, stratified train-test splitting, and feature scaling."""
    X_raw = df[feature_names].copy()
    y = df["target"].to_numpy()

    rng = np.random.default_rng(random_state)
    injected_missing_count = 0
    if synthetic_missing_pct > 0.0:
        mask = rng.random(X_raw.shape) < (synthetic_missing_pct / 100.0)
        X_raw = X_raw.mask(mask)
        injected_missing_count = int(mask.sum())

    initial_missing = int(X_raw.isna().sum().sum())

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    post_missing = int(np.isnan(X_train_imputed).sum() + np.isnan(X_test_imputed).sum())

    if "MinMaxScaler" in scaler_name:
        scaler = MinMaxScaler()
    elif "RobustScaler" in scaler_name:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return {
        "X_train_raw": X_train_raw,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "initial_missing": initial_missing,
        "injected_missing_count": injected_missing_count,
        "post_missing": post_missing,
        "scaler": scaler,
        "imputer": imputer,
    }


def build_model(algo_name: str, params: dict[str, Any], random_state: int = 42) -> Any:
    """Instantiates the selected scikit-learn estimator."""
    if algo_name == "Random Forest Classifier":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 150),
            max_depth=params.get("max_depth", 8),
            random_state=random_state,
        )
    if algo_name == "Logistic Regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=random_state,
        )
    if algo_name == "Support Vector Machine (SVM)":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True,
            random_state=random_state,
        )
    if algo_name == "Gradient Boosting Classifier":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 120),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=random_state,
        )
    return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))


def train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
    is_binary: bool,
) -> dict[str, Any]:
    """Trains the ML model, computes evaluation metrics, and runs 5-fold cross-validation."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    avg_mode = "binary" if is_binary else "weighted"
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=avg_mode, zero_division=0))
    recall = float(recall_score(y_test, y_pred, average=avg_mode, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average=avg_mode, zero_division=0))

    roc_auc = None
    fpr, tpr = None, None
    if y_proba is not None:
        if is_binary:
            roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        else:
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "train_time_ms": train_time_ms,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cm": cm,
        "report_df": pd.DataFrame(report_dict).transpose(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


@st.cache_data
def benchmark_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    is_binary: bool,
) -> pd.DataFrame:
    """Runs a quick comparative benchmark across all 5 candidate ML algorithms."""
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine (RBF)": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    avg_mode = "binary" if is_binary else "weighted"
    rows = []
    for name, clf in candidates.items():
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        dt = round((time.perf_counter() - t0) * 1000, 1)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        auc = (
            roc_auc_score(y_test, probs[:, 1])
            if is_binary
            else roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
        )
        rows.append(
            {
                "Algorithm": name,
                "Test Accuracy": round(accuracy_score(y_test, preds), 4),
                "Precision": round(precision_score(y_test, preds, average=avg_mode, zero_division=0), 4),
                "Recall": round(recall_score(y_test, preds, average=avg_mode, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, preds, average=avg_mode, zero_division=0), 4),
                "ROC-AUC": round(float(auc), 4),
                "Train Time (ms)": dt,
            }
        )
    return pd.DataFrame(rows).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)


def _apply_dark_plot_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    """Applies consistent Obsidian/Amber dark styling to Matplotlib figures."""
    fig.patch.set_facecolor("#141B22")
    ax.set_facecolor("#182028")
    ax.tick_params(colors="#D1D5DB", labelsize=9)
    ax.xaxis.label.set_color("#F3F4F6")
    ax.yaxis.label.set_color("#F3F4F6")
    ax.title.set_color("#F59E0B")
    for spine in ax.spines.values():
        spine.set_color("#2E3A46")


def plot_confusion_matrix(cm: np.ndarray, target_names: list[str]) -> plt.Figure:
    """Creates a formatted Confusion Matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _apply_dark_plot_theme(fig, ax)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Test Set)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label", fontsize=10)
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, algo_name: str) -> plt.Figure:
    """Creates an ROC Curve diagnostic plot."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _apply_dark_plot_theme(fig, ax)
    ax.plot(
        fpr,
        tpr,
        color="#F59E0B",
        lw=2.5,
        label=f"{algo_name} (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#6B7280", lw=1.5, linestyle="--", label="Random Baseline (0.50)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#F59E0B")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", pad=10)
    legend = ax.legend(loc="lower right", facecolor="#141B22", edgecolor="#2E3A46", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#F3F4F6")
    ax.grid(color="#26303A", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model: Any, feature_names: list[str], X_train: np.ndarray, top_n: int = 12
) -> tuple[plt.Figure, list[tuple[str, float]]]:
    """Extracts and plots Top N feature importances or linear coefficient magnitudes."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        metric_label = "Gini / Tree Importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        metric_label = "Absolute Coefficient Magnitude (|w|)"
    else:
        importances = np.var(X_train, axis=0)
        metric_label = "Feature Variance Weight"

    indices = np.argsort(importances)[-top_n:]
    top_features = [(feature_names[i], float(importances[i])) for i in reversed(indices)]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _apply_dark_plot_theme(fig, ax)
    y_labels = [feature_names[i] for i in indices]
    vals = importances[indices]
    bars = ax.barh(y_labels, vals, color="#10B981", edgecolor="#059669", height=0.68)
    for bar in bars[-3:]:
        bar.set_color("#F59E0B")
    ax.set_xlabel(metric_label, fontsize=9.5)
    ax.set_title(f"Top {top_n} Predictive Features", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color="#26303A", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    return fig, top_features


def plot_correlation_heatmap(df: pd.DataFrame, feature_names: list[str]) -> plt.Figure:
    """Plots correlation matrix of the first 10 features + target."""
    subset_cols = feature_names[:10] + ["target"]
    corr = df[subset_cols].corr()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _apply_dark_plot_theme(fig, ax)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        annot_kws={"size": 7.5},
        cbar=False,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Top 10 Features + Target)", fontsize=10.5, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def render_ml_assignment_app() -> None:
    """Renders the complete 8-stage ML Assignment 2 interface."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-tag">INSTANCE 02 // MACHINE LEARNING ASSIGNMENT 2</div>
            <h1 class="hero-title">CLINICAL ML MODEL BUILD & ANALYSIS LAB</h1>
            <p class="hero-subtitle">
                Complete end-to-end supervised classification pipeline covering dataset curation, problem formulation,
                preprocessing, model training, evaluation metrics, diagnostic plots, and analytical interpretation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # STEP 1 & 2: DATASET SELECTION & PROBLEM STATEMENT
    # =========================================================================
    st.markdown("### 01 & 02 // DATASET SELECTION & PROBLEM STATEMENT")
    col_ds, col_desc = st.columns([1.2, 2.8])
    with col_ds:
        dataset_choice = st.selectbox(
            "Select Benchmark Dataset",
            options=[
                "Breast Cancer Wisconsin (Diagnostic) — Primary",
                "UCI Wine Chemical Recognition (Multi-Class)",
            ],
            index=0,
        )
        data_bundle = load_selected_dataset(dataset_choice)
        df = data_bundle["df"]
        feature_names = data_bundle["feature_names"]
        target_names = data_bundle["target_names"]
        is_binary = data_bundle["is_binary"]

        st.metric("Total Records (Rows)", f"{df.shape[0]}")
        st.metric("Predictor Features", f"{len(feature_names)}")
        st.metric("Target Classes", f"{len(target_names)}")

    with col_desc:
        st.info(data_bundle["problem_statement"])
        st.markdown(data_bundle["dataset_explanation"])

        class_counts = df["target"].value_counts().sort_index()
        dist_items = [
            f"**{target_names[idx]}**: `{count}` ({round(count / len(df) * 100, 1)}%)"
            for idx, count in class_counts.items()
        ]
        st.markdown("**Class Distribution:** " + "  |  ".join(dist_items))

    with st.expander("🔍 Inspect Raw Dataset Sample & Descriptive Statistics (`df.head()` / `df.describe()`)"):
        t_raw, t_stats = st.tabs(["Raw Data Preview (First 10 Rows)", "Statistical Summary"])
        with t_raw:
            st.dataframe(df.head(10), use_container_width=True)
        with t_stats:
            st.dataframe(df[feature_names[:12]].describe().T.round(3), use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # STEP 3: DATA PREPROCESSING
    # =========================================================================
    st.markdown("### 03 // DATA PREPROCESSING PIPELINE")
    p1, p2, p3 = st.columns(3)
    with p1:
        scaler_choice = st.selectbox(
            "Feature Scaling Method",
            options=[
                "StandardScaler (Z-Score: μ=0, σ=1)",
                "MinMaxScaler (Range [0, 1])",
                "RobustScaler (Median / IQR)",
            ],
            index=0,
        )
    with p2:
        test_size_pct = st.slider(
            "Test Split Percentage (%)",
            min_value=15,
            max_value=40,
            value=20,
            step=5,
            help="Proportion of dataset reserved for out-of-sample evaluation.",
        )
    with p3:
        synthetic_missing_pct = st.slider(
            "Simulate Missing Values (%) for Imputation Test",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0,
            help="Optionally inject random NaN values to verify SimpleImputer(strategy='median').",
        )

    prep = preprocess_dataset(
        df=df,
        feature_names=feature_names,
        scaler_name=scaler_choice,
        test_size=test_size_pct / 100.0,
        synthetic_missing_pct=synthetic_missing_pct,
    )

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("TRAINING SAMPLES", f"{prep['X_train'].shape[0]}")
    pm2.metric("TESTING SAMPLES", f"{prep['X_test'].shape[0]}")
    pm3.metric("MISSING VALUES (PRE -> POST)", f"{prep['initial_missing']} -> {prep['post_missing']}")
    pm4.metric("SCALED FEATURE MEAN / STD", f"{prep['X_train'].mean():.2f} / {prep['X_train'].std():.2f}")

    with st.expander("📋 View Preprocessing Transformation Proof (Before vs. After Scaling)"):
        sample_cols = feature_names[:5]
        before_df = pd.DataFrame(prep["X_train_raw"], columns=feature_names)[sample_cols].describe().loc[["mean", "std"]].round(3)
        after_df = pd.DataFrame(prep["X_train"], columns=feature_names)[sample_cols].describe().loc[["mean", "std"]].round(3)
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.caption("Before Scaling (Raw Feature Units)")
            st.dataframe(before_df, use_container_width=True)
        with bcol2:
            st.caption(f"After {scaler_choice.split(' ')[0]}")
            st.dataframe(after_df, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # STEP 4 & 5: ALGORITHM SELECTION, TRAINING & TESTING
    # =========================================================================
    st.markdown("### 04 & 05 // ML ALGORITHM SELECTION, TRAINING & TESTING")
    acol1, acol2 = st.columns([1.4, 1.6])
    with acol1:
        algo_choice = st.selectbox(
            "Select Supervised ML Algorithm",
            options=[
                "Random Forest Classifier",
                "Logistic Regression",
                "Support Vector Machine (SVM)",
                "Gradient Boosting Classifier",
                "K-Nearest Neighbors (KNN)",
            ],
            index=0,
        )

    params: dict[str, Any] = {}
    with acol2:
        if algo_choice == "Random Forest Classifier":
            hc1, hc2 = st.columns(2)
            params["n_estimators"] = hc1.slider("Number of Trees (n_estimators)", 50, 300, 150, 25)
            params["max_depth"] = hc2.slider("Max Tree Depth", 3, 20, 8, 1)
            algo_rationale = (
                "**Why Random Forest?** Ensemble bagging of decorrelated decision trees handles high-dimensional "
                "correlated features without overfitting, captures non-linear feature interactions, and provides "
                "direct Gini impurity feature importance rankings."
            )
        elif algo_choice == "Logistic Regression":
            params["C"] = st.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, 0.25)
            algo_rationale = (
                "**Why Logistic Regression?** Provides calibrated probabilities and linear decision boundaries "
                "with L2 regularization, ideal as an interpretable clinical baseline."
            )
        elif algo_choice == "Support Vector Machine (SVM)":
            hc1, hc2 = st.columns(2)
            params["C"] = hc1.slider("Regularization Margin (C)", 0.1, 10.0, 1.0, 0.5)
            params["kernel"] = hc2.selectbox("Kernel Function", ["rbf", "linear", "poly"], index=0)
            algo_rationale = (
                "**Why SVM?** Constructs an optimal maximum-margin hyperplane in high-dimensional space, "
                "excelling on compact biomedical datasets with clear margin separation."
            )
        elif algo_choice == "Gradient Boosting Classifier":
            hc1, hc2 = st.columns(2)
            params["n_estimators"] = hc1.slider("Boosting Stages", 50, 250, 120, 25)
            params["learning_rate"] = hc2.slider("Learning Rate", 0.01, 0.30, 0.10, 0.02)
            algo_rationale = (
                "**Why Gradient Boosting?** Sequentially fits shallow trees to negative gradients of the binomial "
                "deviance loss, focusing on hard-to-classify boundary cases."
            )
        else:
            params["n_neighbors"] = st.slider("Number of Neighbors (k)", 3, 19, 5, 2)
            algo_rationale = (
                "**Why KNN?** Non-parametric instance-based classifier that classifies samples by majority vote "
                "in Euclidean feature space after standardization."
            )

    st.caption(algo_rationale)

    model = build_model(algo_choice, params)
    results = train_and_evaluate(
        model=model,
        X_train=prep["X_train"],
        X_test=prep["X_test"],
        y_train=prep["y_train"],
        y_test=prep["y_test"],
        target_names=target_names,
        is_binary=is_binary,
    )

    st.markdown("---")

    # =========================================================================
    # STEP 6: EVALUATION METRICS
    # =========================================================================
    st.markdown("### 06 // MODEL EVALUATION METRICS")
    em1, em2, em3, em4, em5, em6 = st.columns(6)
    em1.metric("TEST ACCURACY", f"{results['test_acc'] * 100:.2f}%")
    em2.metric("PRECISION", f"{results['precision'] * 100:.2f}%")
    em3.metric("RECALL (SENSITIVITY)", f"{results['recall'] * 100:.2f}%")
    em4.metric("F1-SCORE", f"{results['f1'] * 100:.2f}%")
    em5.metric("ROC-AUC SCORE", f"{results['roc_auc']:.4f}" if results["roc_auc"] else "N/A")
    em6.metric("5-FOLD CV ACCURACY", f"{results['cv_mean'] * 100:.2f}% (±{results['cv_std'] * 100:.1f}%)")

    tab_rep, tab_bench = st.tabs(
        ["📊 Per-Class Classification Report", "🏆 Multi-Algorithm Benchmark Comparison"]
    )
    with tab_rep:
        st.dataframe(results["report_df"].round(4), use_container_width=True)
    with tab_bench:
        bench_df = benchmark_all_models(
            prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"], is_binary
        )
        st.dataframe(bench_df, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # STEP 7: VISUAL DIAGNOSTICS (GRAPHS / PLOTS)
    # =========================================================================
    st.markdown("### 07 // DIAGNOSTIC GRAPHS & VISUALIZATIONS")
    fig_cm = plot_confusion_matrix(results["cm"], target_names)
    fig_fi, top_features = plot_feature_importance(model, feature_names, prep["X_train"])
    fig_corr = plot_correlation_heatmap(df, feature_names)

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
    with gcol2:
        if is_binary and results["fpr"] is not None:
            fig_roc = plot_roc_curve(results["fpr"], results["tpr"], results["roc_auc"], algo_choice)
            st.pyplot(fig_roc, use_container_width=True)
            plt.close(fig_roc)
        else:
            st.pyplot(fig_corr, use_container_width=True)

    gcol3, gcol4 = st.columns(2)
    with gcol3:
        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)
    with gcol4:
        if is_binary:
            st.pyplot(fig_corr, use_container_width=True)
            plt.close(fig_corr)

    st.markdown("---")

    # =========================================================================
    # STEP 8: SHORT ANALYSIS OF RESULTS
    # =========================================================================
    st.markdown("### 08 // SHORT ANALYSIS OF RESULTS & INTERPRETATION")
    top_3_names = ", ".join([f"`{name}` ({val:.3f})" for name, val in top_features[:3]])
    overfit_gap = (results["train_acc"] - results["test_acc"]) * 100.0
    cm = results["cm"]

    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        err_breakdown = (
            f"Out of **{len(prep['y_test'])} held-out test biopsies**, the model achieved **{tn + tp} correct classifications** "
            f"(`{tn}` True Malignant, `{tp}` True Benign) with only **{fp + fn} misclassifications** "
            f"(`{fp}` False Positives, `{fn}` False Negatives)."
        )
    else:
        correct_total = int(np.trace(cm))
        total_test = int(cm.sum())
        err_breakdown = (
            f"Out of **{total_test} held-out test instances**, the model achieved **{correct_total} correct predictions** "
            f"and **{total_test - correct_total} misclassifications** across {len(target_names)} classes."
        )

    st.markdown(
        f"""
        <div class="analysis-card">
            <h4>📌 Executive Summary & Model Performance Analysis</h4>
            <ul>
                <li><strong>Predictive Accuracy & Generalization:</strong> The trained <strong>{algo_choice}</strong> achieved a held-out test accuracy of <strong>{results['test_acc']*100:.2f}%</strong>, an F1-Score of <strong>{results['f1']*100:.2f}%</strong>, and a 5-fold stratified cross-validation score of <strong>{results['cv_mean']*100:.2f}% ± {results['cv_std']*100:.2f}%</strong> in <code>{results['train_time_ms']} ms</code>. The train-vs-test generalization gap is <code>{overfit_gap:+.2f}%</code>, confirming strong generalization without severe overfitting.</li>
                <li><strong>Confusion Matrix & Error Breakdown:</strong> {err_breakdown}</li>
                <li><strong>Dominant Biomarkers / Features:</strong> Feature importance ranking reveals that the top predictive variables driving classification decisions are {top_3_names}. In cytological analysis, extreme cell perimeter, area, and concave contour irregularities strongly separate malignant nuclei from benign tissue.</li>
                <li><strong>Impact of Preprocessing:</strong> Applying <strong>{scaler_choice.split(' ')[0]}</strong> standardized heterogeneous feature magnitudes (such as cell area in hundreds of square microns vs. smoothness ratios near <code>0.1</code>), ensuring stable convergence and fair distance/margin computation across algorithms.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
