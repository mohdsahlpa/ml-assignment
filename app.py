"""Main Entrypoint: Streamlit Assignment Project."""

import streamlit as st

from modules.face_detection import render_face_detection_app
from modules.ml_assignment import render_ml_assignment_app

st.set_page_config(
    page_title="ML Assignment",
    layout="wide",
)


def render_home() -> None:
    """Renders a clean, simple selection screen for the two assignment applications."""
    st.title("Machine Learning Assignment")
    st.write("Select one of the two applications below:")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Face Detection Mobile Application")
        st.write(
            "Detect faces and eyes in images using OpenCV Haar Cascades. "
            "Supports mobile/webcam camera capture, image file upload, and built-in sample images."
        )
        if st.button("Open Face Detection Application", use_container_width=True):
            st.session_state["page"] = "1. Face Detection Application"
            st.rerun()

    with col2:
        st.subheader("2. ML Assignment 2: Build and Analyse an ML Model")
        st.write(
            "End-to-end machine learning workflow: dataset selection, problem statement, "
            "data preprocessing, model training and testing, evaluation metrics, plots, and result analysis."
        )
        if st.button("Open ML Assignment 2", use_container_width=True):
            st.session_state["page"] = "2. ML Assignment 2"
            st.rerun()


def main() -> None:
    """Main navigation router."""
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    options = [
        "Home",
        "1. Face Detection Application",
        "2. ML Assignment 2",
    ]

    with st.sidebar:
        st.header("Navigation")
        selected = st.radio(
            "Go to:",
            options=options,
            index=options.index(st.session_state["page"]),
            key="nav_radio",
        )
        if selected != st.session_state["page"]:
            st.session_state["page"] = selected
            st.rerun()

    if st.session_state["page"] == "1. Face Detection Application":
        render_face_detection_app()
    elif st.session_state["page"] == "2. ML Assignment 2":
        render_ml_assignment_app()
    else:
        render_home()


if __name__ == "__main__":
    main()
