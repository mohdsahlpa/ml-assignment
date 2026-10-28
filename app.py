"""Main Entrypoint: Unified Streamlit Dashboard for Face Detection Mobile App & ML Assignment 2."""

import streamlit as st

from modules.face_detection import render_face_detection_app
from modules.ml_assignment import render_ml_assignment_app

st.set_page_config(
    page_title="ML & CV Dual-Instance Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    """Injects custom geometric brutalist telemetry CSS (Zero-Purple, Signal Amber & Emerald palette)."""
    st.markdown(
        """
        <style>
        /* Sharp Geometric Brutalist Telemetry Theme */
        :root {
            --bg-primary: #0F1419;
            --bg-panel: #182028;
            --border-raw: #2E3A46;
            --accent-amber: #F59E0B;
            --accent-emerald: #10B981;
            --accent-coral: #EF4444;
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
        }

        .hero-banner {
            background: linear-gradient(135deg, #141B22 0%, #1C2632 100%);
            border-left: 5px solid #F59E0B;
            border-top: 1px solid #2E3A46;
            border-right: 1px solid #2E3A46;
            border-bottom: 1px solid #2E3A46;
            border-radius: 2px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
        }

        .hero-tag {
            font-family: monospace;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            color: #F59E0B;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: 2.0rem;
            font-weight: 800;
            color: #F3F4F6;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 0.98rem;
            color: #D1D5DB;
            margin: 0;
            line-height: 1.55;
        }

        .instance-card {
            background-color: #182028;
            border: 1px solid #2E3A46;
            border-top: 4px solid #F59E0B;
            border-radius: 2px;
            padding: 1.5rem;
            height: 100%;
            transition: transform 0.18s ease-out, border-color 0.18s ease-out;
        }

        .instance-card:hover {
            transform: translateY(-3px);
            border-color: #F59E0B;
        }

        .instance-card-emerald {
            border-top: 4px solid #10B981;
        }

        .instance-card-emerald:hover {
            border-color: #10B981;
        }

        .card-badge {
            display: inline-block;
            font-family: monospace;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            background-color: #24303C;
            color: #F59E0B;
            border-radius: 2px;
            margin-bottom: 0.75rem;
        }

        .card-badge-emerald {
            color: #10B981;
        }

        .analysis-card {
            background-color: #161F27;
            border: 1px solid #2E3A46;
            border-left: 4px solid #10B981;
            border-radius: 2px;
            padding: 1.35rem 1.6rem;
            line-height: 1.65;
        }

        .analysis-card h4 {
            color: #10B981;
            margin-top: 0;
            margin-bottom: 0.75rem;
        }

        div[data-testid="stMetric"] {
            background-color: #182028;
            border: 1px solid #2E3A46;
            border-radius: 2px;
            padding: 0.85rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_launchpad() -> None:
    """Renders the main Landing Dashboard allowing the user to choose between both instances."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-tag">UNIFIED STREAMLIT CLOUD WORKSPACE // DUAL-INSTANCE PORTAL</div>
            <h1 class="hero-title">MACHINE LEARNING & COMPUTER VISION COMMAND SUITE</h1>
            <p class="hero-subtitle">
                Select an application instance below to launch its dedicated workspace. Both applications run natively
                within this unified Streamlit Cloud deployment and support mobile & desktop browsers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="instance-card">
                <span class="card-badge">INSTANCE 01 // MOBILE CV</span>
                <h3>📱 Face Detection Mobile Application</h3>
                <p style="color: #D1D5DB; font-size: 0.94rem;">
                    Real-time mobile-friendly biometric detection studio powered by OpenCV Haar Cascades.
                </p>
                <ul style="color: #9CA3AF; font-size: 0.88rem; line-height: 1.6;">
                    <li><strong>Mobile Camera Integration:</strong> Direct phone camera snapshot capture (<code>st.camera_input</code>) & image upload.</li>
                    <li><strong>Multi-Cascade Sensor:</strong> Simultaneous detection of frontal faces, ocular regions (eyes), and smiles.</li>
                    <li><strong>Interactive Tuning & Privacy:</strong> Adjust scale pyramid, neighbor thresholds, and face blur/pixelation filters.</li>
                    <li><strong>ROI Cropper & Export:</strong> Inspect cropped face bounding boxes and download annotated PNG frames.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🚀 LAUNCH INSTANCE 01: FACE DETECTION APP", use_container_width=True, type="primary"):
            st.session_state["active_instance"] = "Instance 1: Face Detection Mobile App"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="instance-card instance-card-emerald">
                <span class="card-badge card-badge-emerald">INSTANCE 02 // ML ASSIGNMENT 2</span>
                <h3>📊 ML Assignment 2: Build & Analyse an ML Model</h3>
                <p style="color: #D1D5DB; font-size: 0.94rem;">
                    End-to-end supervised Machine Learning workflow fulfilling all 8 assignment evaluation criteria.
                </p>
                <ul style="color: #9CA3AF; font-size: 0.88rem; line-height: 1.6;">
                    <li><strong>Dataset & Problem Formulation:</strong> Breast Cancer Wisconsin (Diagnostic) & UCI Wine datasets with statistical EDA.</li>
                    <li><strong>Data Preprocessing:</strong> Median imputation, StandardScaler / MinMaxScaler / RobustScaler, and stratified splits.</li>
                    <li><strong>Model Training & Evaluation:</strong> Random Forest, Logistic Regression, SVM, Gradient Boosting & KNN + 5-Fold CV.</li>
                    <li><strong>Visual Plots & Analysis:</strong> Confusion Matrix, ROC-AUC Curve, Feature Importance, Correlation Heatmap & written analysis.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("📈 LAUNCH INSTANCE 02: ML ASSIGNMENT 2 LAB", use_container_width=True, type="primary"):
            st.session_state["active_instance"] = "Instance 2: ML Assignment 2 (Build & Analyse Model)"
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚡ QUICK ARCHITECTURE OVERVIEW")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ACTIVE INSTANCES", "2 Apps in 1")
    k2.metric("CV ENGINE", "OpenCV Headless")
    k3.metric("ML ALGORITHMS", "5 Estimators")
    k4.metric("CLOUD TARGET", "Streamlit Free Tier")


def main() -> None:
    """Main router controlling dashboard navigation between Launchpad and the two app instances."""
    inject_custom_styles()

    if "active_instance" not in st.session_state:
        st.session_state["active_instance"] = "🏠 Central Launchpad (Choose App)"

    nav_options = [
        "🏠 Central Launchpad (Choose App)",
        "Instance 1: Face Detection Mobile App",
        "Instance 2: ML Assignment 2 (Build & Analyse Model)",
    ]

    with st.sidebar:
        st.markdown("### ⚡ WORKSPACE SWITCHER")
        selected = st.radio(
            "Select Active Application:",
            options=nav_options,
            index=nav_options.index(st.session_state["active_instance"]),
            key="workspace_switcher",
        )
        if selected != st.session_state["active_instance"]:
            st.session_state["active_instance"] = selected
            st.rerun()

        st.markdown("---")
        st.caption(
            "**Deployment Ready:** Configured with `opencv-python-headless` and `scikit-learn` for one-click deployment on Streamlit Community Cloud via GitHub."
        )
        if (
            st.session_state["active_instance"] != "🏠 Central Launchpad (Choose App)"
            and st.button("⬅️ Back to Launchpad", use_container_width=True)
        ):
            st.session_state["active_instance"] = "🏠 Central Launchpad (Choose App)"
            st.rerun()

    current = st.session_state["active_instance"]
    if current == "Instance 1: Face Detection Mobile App":
        render_face_detection_app()
    elif current == "Instance 2: ML Assignment 2 (Build & Analyse Model)":
        render_ml_assignment_app()
    else:
        render_launchpad()


if __name__ == "__main__":
    main()
