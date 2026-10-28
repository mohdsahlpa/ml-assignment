"""Automated verification and unit tests for Face Detection and ML Assignment 2 modules."""

import unittest

from modules.face_detection import detect_faces_and_features
from modules.ml_assignment import (
    benchmark_all_models,
    build_model,
    load_selected_dataset,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_roc_curve,
    preprocess_dataset,
    train_and_evaluate,
)
from utils.sample_generator import generate_sample_portrait


class TestDualStreamlitSuite(unittest.TestCase):
    """Tests both Instance 1 (Face Detection) and Instance 2 (ML Assignment 2)."""

    def test_sample_portrait_generation_and_detection(self) -> None:
        """Verifies sample portrait shape and face detection pipeline execution."""
        single_img = generate_sample_portrait("single")
        self.assertEqual(single_img.shape[2], 3)
        annotated, detections, elapsed_ms = detect_faces_and_features(
            single_img,
            scale_factor=1.1,
            min_neighbors=3,
            min_size=30,
            detect_eyes=True,
            detect_smiles=True,
        )
        self.assertEqual(annotated.shape, single_img.shape)
        self.assertGreaterEqual(len(detections), 1)
        self.assertGreater(elapsed_ms, 0.0)

    def test_ml_pipeline_breast_cancer(self) -> None:
        """Verifies dataset loading, preprocessing, training, metrics, and plots."""
        bundle = load_selected_dataset("Breast Cancer Wisconsin (Diagnostic) — Primary")
        df = bundle["df"]
        features = bundle["feature_names"]
        targets = bundle["target_names"]

        self.assertEqual(df.shape[0], 569)
        self.assertEqual(len(features), 30)

        prep = preprocess_dataset(
            df=df,
            feature_names=features,
            scaler_name="StandardScaler (Z-Score)",
            test_size=0.20,
            synthetic_missing_pct=2.0,
        )
        self.assertEqual(prep["post_missing"], 0)

        model = build_model("Random Forest Classifier", {"n_estimators": 100, "max_depth": 6})
        results = train_and_evaluate(
            model=model,
            X_train=prep["X_train"],
            X_test=prep["X_test"],
            y_train=prep["y_train"],
            y_test=prep["y_test"],
            target_names=targets,
            is_binary=True,
        )
        self.assertGreater(results["test_acc"], 0.90)
        self.assertGreater(results["roc_auc"], 0.95)

        fig_cm = plot_confusion_matrix(results["cm"], targets)
        self.assertIsNotNone(fig_cm)
        fig_roc = plot_roc_curve(results["fpr"], results["tpr"], results["roc_auc"], "Random Forest")
        self.assertIsNotNone(fig_roc)
        fig_fi, top_feats = plot_feature_importance(model, features, prep["X_train"])
        self.assertIsNotNone(fig_fi)
        self.assertEqual(len(top_feats), 12)
        fig_corr = plot_correlation_heatmap(df, features)
        self.assertIsNotNone(fig_corr)

        bench_df = benchmark_all_models(
            prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"], True
        )
        self.assertEqual(len(bench_df), 5)


if __name__ == "__main__":
    unittest.main()
