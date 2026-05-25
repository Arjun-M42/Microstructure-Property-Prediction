from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def relaunch_with_streamlit_when_needed() -> None:
    if __name__ == "__main__" and get_script_run_ctx() is None:
        app_path = Path(__file__).resolve()
        raise SystemExit(
            subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
        )


relaunch_with_streamlit_when_needed()

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import (  # noqa: E402
    CHECKPOINT_PATH,
    TRAIN_IMAGE_DIR,
    predict_image_arrays,
    predict_structure,
)

st.set_page_config(page_title="Microstructure Kappa Results", layout="wide")

DATA_DIR = PROJECT_ROOT / "data"


def tensor_dataframe(result: dict, prefix: str) -> pd.DataFrame:
    values = result.get(f"{prefix}_tensor")
    if values is None:
        flat_values = result.get(f"{prefix}_values", [])
        return pd.DataFrame(
            {
                "component": list(range(len(flat_values))),
                "tensor_position": [f"kappa[{index // 3},{index % 3}]" for index in range(len(flat_values))],
                prefix: flat_values,
            }
        )
    return pd.DataFrame(values, index=["row 0", "row 1", "row 2"], columns=["col 0", "col 1", "col 2"])


def component_dataframe(result: dict) -> pd.DataFrame:
    predicted = result.get("predicted_values", [])
    actual = result.get("actual_values", [])
    rows = []
    for index, pred_value in enumerate(predicted):
        row = {
            "component": index,
            "tensor_position": f"kappa[{index // 3},{index % 3}]",
            "predicted": pred_value,
        }
        if index < len(actual):
            actual_value = actual[index]
            row["actual"] = actual_value
            row["absolute_error"] = abs(pred_value - actual_value)
        rows.append(row)
    return pd.DataFrame(rows)


def render_result(result: dict) -> None:
    if not result:
        st.info("Run a prediction to see results.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted scalar kappa", f"{result['predicted_scalar_kappa']:.4f}")

    if "actual_scalar_kappa" in result:
        metric_cols[1].metric("Actual scalar kappa", f"{result['actual_scalar_kappa']:.4f}")
        metric_cols[2].metric("MAE", f"{result['mae']:.4f}")
        # metric_cols[3].metric("Accuracy", f"{result['accuracy']:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted tensor")
        st.caption("Rows and columns represent the 3 x 3 kappa tensor. Component 0 is kappa[0,0], component 8 is kappa[2,2].")
        st.dataframe(tensor_dataframe(result, "predicted"), use_container_width=True)
    with right:
        if "actual_tensor" in result:
            st.subheader("Actual tensor")
            st.dataframe(tensor_dataframe(result, "actual"), use_container_width=True)
        else:
            st.subheader("Model details")
            st.info("Select a label file in the sidebar to show actual kappa values beside predictions.")

    st.subheader("Component results")
    st.dataframe(component_dataframe(result), use_container_width=True, hide_index=True)


def decode_uploaded_images(files) -> list[np.ndarray]:
    images = []
    for file in files:
        file.seek(0)
        bytes_array = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(bytes_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            st.warning(f"Could not read {file.name}; skipping it.")
            continue
        images.append(image)
    return images


st.title("Microstructure Property Prediction")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

checkpoint_path = st.sidebar.text_input("Checkpoint", value=str(CHECKPOINT_PATH))
image_dir = st.sidebar.text_input("Image directory", value=str(TRAIN_IMAGE_DIR))
label_options = {
    "No labels": None,
    "Train labels": DATA_DIR / "kappa_train.mat",
    "Validation labels": DATA_DIR / "kappa_val.mat",
    "Test labels": DATA_DIR / "kappa_test.mat",
}
available_label_options = {
    label: path for label, path in label_options.items() if path is None or path.exists()
}
default_label = "Train labels" if "Train labels" in available_label_options else "No labels"
label_choice = st.sidebar.selectbox(
    "Label comparison",
    list(available_label_options.keys()),
    index=list(available_label_options.keys()).index(default_label),
)
label_file = available_label_options[label_choice]

tab_structure, tab_upload, tab_results = st.tabs(
    ["Structure result", "Custom image inference", "Results"]
)

with tab_structure:
    col_input, col_action = st.columns([2, 1])
    with col_input:
        structure_id = st.number_input("Structure number", min_value=1, value=1, step=1)
    with col_action:
        st.write("")
        st.write("")
        run_structure = st.button("Run prediction", type="primary", use_container_width=True)

    if run_structure:
        try:
            result = predict_structure(
                int(structure_id),
                image_dir=image_dir,
                label_file=str(label_file) if label_file else None,
                checkpoint_path=checkpoint_path,
            )
            st.session_state.last_result = result
            render_result(result)
        except Exception as exc:
            st.error(str(exc))

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload one or more slice images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        preview_cols = st.columns(min(4, len(uploaded_files)))
        for index, file in enumerate(uploaded_files[:4]):
            preview_cols[index % len(preview_cols)].image(
                file.getvalue(),
                caption=file.name,
                use_container_width=True,
            )

    if st.button("Predict uploaded images", type="primary", use_container_width=True):
        try:
            images = decode_uploaded_images(uploaded_files or [])
            result = predict_image_arrays(images, checkpoint_path=checkpoint_path)
            st.session_state.last_result = result
            render_result(result)
        except Exception as exc:
            st.error(str(exc))

with tab_results:
    render_result(st.session_state.last_result)
