"""Application 2: ML Assignment 2 — Build and Analyse an ML Model."""

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


@st.cache_data
def load_selected_dataset(dataset_name: str) -> dict[str, Any]:
    """Loads the selected scikit-learn dataset."""
    if "Wine" in dataset_name:
        raw = load_wine()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = [str(t) for t in raw.target_names]
        problem_statement = (
            "Classify wine samples into one of three cultivar classes (Class 0, Class 1, Class 2) "
            "based on 13 chemical analysis measurements such as alcohol content, flavanoids, color intensity, and proline."
        )
        dataset_explanation = (
            "The UCI Wine dataset contains 178 samples and 13 numerical features representing chemical properties "
            "of wines grown in the same region in Italy."
        )
        is_binary = False
    else:
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = ["Malignant (0)", "Benign (1)"]
        problem_statement = (
            "Predict whether a breast tumor is Malignant (0) or Benign (1) based on measurements computed "
            "from digitized images of fine-needle aspirate (FNA) biopsies of breast masses."
        )
        dataset_explanation = (
            "The Breast Cancer Wisconsin (Diagnostic) dataset contains 569 instances (212 Malignant, 357 Benign) "
            "and 30 continuous numerical features describing cell nuclei characteristics such as radius, texture, "
            "perimeter, area, smoothness, compactness, and concavity."
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
    scaler_name: str = "StandardScaler",
    test_size: float = 0.20,
    synthetic_missing_pct: float = 0.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Performs missing value imputation, train/test split, and feature scaling."""
    X_raw = df[feature_names].copy()
    y = df["target"].to_numpy()

    if synthetic_missing_pct > 0.0:
        rng = np.random.default_rng(random_state)
        mask = rng.random(X_raw.shape) < (synthetic_missing_pct / 100.0)
        X_raw = X_raw.mask(mask)

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

    scaler = MinMaxScaler() if "MinMax" in scaler_name else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return {
        "X_train_raw": X_train_raw,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "initial_missing": initial_missing,
        "post_missing": post_missing,
    }


def build_model(algo_name: str, params: dict[str, Any], random_state: int = 42) -> Any:
    """Returns the configured scikit-learn model."""
    if algo_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
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
    if algo_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
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
    """Trains the model and calculates evaluation metrics."""
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
    }


def plot_confusion_matrix(cm: np.ndarray, target_names: list[str]) -> plt.Figure:
    """Plots the Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, algo_name: str) -> plt.Figure:
    """Plots the ROC Curve."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{algo_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model: Any, feature_names: list[str], X_train: np.ndarray, top_n: int = 10
) -> tuple[plt.Figure, list[tuple[str, float]]]:
    """Plots Top N Feature Importances or Coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        xlabel = "Importance Score"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        xlabel = "Absolute Coefficient (|w|)"
    else:
        importances = np.var(X_train, axis=0)
        xlabel = "Feature Variance"

    indices = np.argsort(importances)[-top_n:]
    top_features = [(feature_names[i], float(importances[i])) for i in reversed(indices)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([feature_names[i] for i in indices], importances[indices], color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} Important Features")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig, top_features


def plot_correlation_heatmap(df: pd.DataFrame, feature_names: list[str]) -> plt.Figure:
    """Plots correlation matrix of the first 8 features and target."""
    subset_cols = feature_names[:8] + ["target"]
    corr = df[subset_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap (First 8 Features)")
    fig.tight_layout()
    return fig


def render_ml_assignment_app() -> None:
    """Renders the ML Assignment 2 interface following the exact assignment steps."""
    st.title("2. ML Assignment 2: Build and Analyse an ML Model")

    # 1. Select a suitable dataset
    st.header("1. Select a Suitable Dataset")
    dataset_choice = st.selectbox(
        "Dataset:",
        options=[
            "Breast Cancer Wisconsin (Diagnostic) Dataset",
            "Wine Recognition Dataset",
        ],
    )
    data_bundle = load_selected_dataset(dataset_choice)
    df = data_bundle["df"]
    feature_names = data_bundle["feature_names"]
    target_names = data_bundle["target_names"]
    is_binary = data_bundle["is_binary"]

    # 2. Explain the problem statement and dataset
    st.header("2. Problem Statement and Dataset Explanation")
    st.write(f"**Problem Statement:** {data_bundle['problem_statement']}")
    st.write(f"**Dataset Overview:** {data_bundle['dataset_explanation']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Samples", df.shape[0])
    c2.metric("Number of Features", len(feature_names))
    c3.metric("Target Classes", ", ".join(target_names))

    with st.expander("View Dataset Preview and Summary Statistics"):
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Summary Statistics")
        st.dataframe(df[feature_names[:10]].describe().T.round(3), use_container_width=True)

    # 3. Perform data preprocessing
    st.header("3. Data Preprocessing")
    st.write(
        "Preprocessing steps applied:\n"
        "1. **Missing Value Imputation:** Median imputation (`SimpleImputer(strategy='median')`) to handle any missing entries.\n"
        "2. **Train-Test Split:** Stratified split so class proportions are preserved in both training and test sets.\n"
        "3. **Feature Scaling:** Standardization (`StandardScaler` or `MinMaxScaler`) so features with large numeric ranges do not dominate the model."
    )

    p1, p2 = st.columns(2)
    with p1:
        scaler_choice = st.selectbox("Feature Scaling:", ["StandardScaler (Zero Mean, Unit Variance)", "MinMaxScaler (0 to 1)"])
    with p2:
        test_size_pct = st.slider("Test Set Size (%):", 15, 40, 20, 5)

    prep = preprocess_dataset(
        df=df,
        feature_names=feature_names,
        scaler_name=scaler_choice,
        test_size=test_size_pct / 100.0,
    )

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Training Set Shape", f"{prep['X_train'].shape[0]} x {prep['X_train'].shape[1]}")
    pc2.metric("Testing Set Shape", f"{prep['X_test'].shape[0]} x {prep['X_test'].shape[1]}")
    pc3.metric("Missing Values After Imputation", prep["post_missing"])

    # 4. Select an appropriate ML algorithm
    st.header("4. Select an Appropriate ML Algorithm")
    ac1, ac2 = st.columns(2)
    with ac1:
        algo_choice = st.selectbox(
            "Algorithm:",
            [
                "Random Forest",
                "Logistic Regression",
                "Support Vector Machine (SVM)",
                "Gradient Boosting",
                "K-Nearest Neighbors (KNN)",
            ],
        )

    params: dict[str, Any] = {}
    with ac2:
        if algo_choice == "Random Forest":
            params["n_estimators"] = st.slider("Number of Trees (n_estimators):", 50, 250, 100, 25)
            params["max_depth"] = st.slider("Max Depth:", 3, 20, 8, 1)
        elif algo_choice == "Logistic Regression":
            params["C"] = st.slider("Regularization Parameter (C):", 0.1, 10.0, 1.0, 0.1)
        elif algo_choice == "Support Vector Machine (SVM)":
            params["C"] = st.slider("Regularization Parameter (C):", 0.1, 10.0, 1.0, 0.1)
            params["kernel"] = st.selectbox("Kernel:", ["rbf", "linear"])
        elif algo_choice == "Gradient Boosting":
            params["n_estimators"] = st.slider("Number of Trees:", 50, 200, 100, 25)
            params["learning_rate"] = st.slider("Learning Rate:", 0.01, 0.30, 0.10, 0.01)
        else:
            params["n_neighbors"] = st.slider("Neighbors (k):", 3, 15, 5, 2)

    # 5. Train and test the model
    st.header("5. Train and Test the Model")
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
    st.write(
        f"The **{algo_choice}** model was trained on **{prep['X_train'].shape[0]} samples** "
        f"(training time: {results['train_time_ms']} ms) and tested on **{prep['X_test'].shape[0]} unseen samples**."
    )

    # 6. Calculate suitable evaluation metrics
    st.header("6. Evaluation Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test Accuracy", f"{results['test_acc'] * 100:.2f}%")
    m2.metric("Precision", f"{results['precision'] * 100:.2f}%")
    m3.metric("Recall", f"{results['recall'] * 100:.2f}%")
    m4.metric("F1-Score", f"{results['f1'] * 100:.2f}%")
    m5.metric("ROC-AUC", f"{results['roc_auc']:.4f}" if results["roc_auc"] else "N/A")

    st.subheader("Classification Report")
    st.dataframe(results["report_df"].round(4), use_container_width=True)

    # 7. Display the results using graphs/plots
    st.header("7. Results Using Graphs and Plots")
    fig_cm = plot_confusion_matrix(results["cm"], target_names)
    fig_fi, top_features = plot_feature_importance(model, feature_names, prep["X_train"], top_n=10)
    fig_corr = plot_correlation_heatmap(df, feature_names)

    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with g2:
        if is_binary and results["fpr"] is not None:
            fig_roc = plot_roc_curve(results["fpr"], results["tpr"], results["roc_auc"], algo_choice)
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.pyplot(fig_corr)

    g3, g4 = st.columns(2)
    with g3:
        st.pyplot(fig_fi)
        plt.close(fig_fi)
    with g4:
        if is_binary:
            st.pyplot(fig_corr)
            plt.close(fig_corr)

    # 8. Give a short analysis of the results
    st.header("8. Short Analysis of Results")
    top_3 = ", ".join([f"{name} ({val:.3f})" for name, val in top_features[:3]])
    cm = results["cm"]
    correct = int(np.trace(cm))
    total = int(cm.sum())
    errors = total - correct

    st.write(
        f"- **Model Performance:** The **{algo_choice}** achieved a test accuracy of **{results['test_acc']*100:.2f}%**, "
        f"a precision of **{results['precision']*100:.2f}%**, a recall of **{results['recall']*100:.2f}%**, and an F1-score of **{results['f1']*100:.2f}%**. "
        f"Across 5-fold cross-validation on the training set, the mean accuracy was **{results['cv_mean']*100:.2f}% (+/- {results['cv_std']*100:.2f}%)**.\n"
        f"- **Confusion Matrix Breakdown:** Out of **{total}** test samples, the model classified **{correct}** samples correctly and made **{errors}** errors.\n"
        f"- **Key Features:** The most influential features for classification were **{top_3}**.\n"
        f"- **Conclusion:** Feature standardization combined with {algo_choice} produced high accuracy and strong separation between classes with minimal overfitting (training accuracy: {results['train_acc']*100:.2f}% vs. test accuracy: {results['test_acc']*100:.2f}%)."
    )
