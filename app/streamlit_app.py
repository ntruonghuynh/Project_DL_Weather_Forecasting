"""Streamlit HTTP client for the Jena forecasting FastAPI service."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def api_get(base_url: str, path: str, timeout: float = 10.0) -> dict:
    with urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def api_predict(base_url: str, payload: dict, timeout: float = 60.0) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_upload(uploaded_file) -> dict:
    payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    required = {"timestamps", "observations"}
    optional = {"future_timestamps", "future_actual"}
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or set(payload) - required - optional
    ):
        raise ValueError(
            "JSON requires timestamps and observations; optional future_timestamps/future_actual"
        )
    return payload


def main() -> None:
    """Render a demo that delegates all inference to FastAPI."""
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Jena 72-hour Forecast", layout="wide")
    st.title("Jena temperature forecast")
    st.caption("168 hours of observations → FastAPI → 72-hour temperature forecast")
    default_url = os.getenv("JENA_API_URL", "http://127.0.0.1:8000")
    base_url = st.sidebar.text_input("FastAPI URL", default_url)
    try:
        health = api_get(base_url, "/health")
        if health.get("status") == "ready":
            st.sidebar.success("API ready")
            st.sidebar.json(api_get(base_url, "/model-info"))
        else:
            st.sidebar.warning(health.get("detail", "API not ready"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        st.sidebar.error(f"Cannot reach API: {error}")

    uploaded = st.file_uploader(
        "Upload inference JSON",
        type=["json"],
        help="Object with timestamps[168] and observations[168] keyed by feature schema.",
    )
    if uploaded is None:
        st.info("Upload a real 168-hour input payload to run the versioned model bundle.")
        return
    try:
        payload = _parse_upload(uploaded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Invalid input file: {error}")
        return

    history = pd.DataFrame(payload["observations"], index=pd.to_datetime(payload["timestamps"]))
    st.subheader("Input history")
    target_candidates = [name for name in history if "degC" in name]
    if target_candidates:
        st.line_chart(history[target_candidates[0]])
    st.dataframe(history.tail(24), use_container_width=True)

    if st.button("Forecast next 72 hours", type="primary"):
        try:
            api_payload = {
                "timestamps": payload["timestamps"],
                "observations": payload["observations"],
            }
            result = api_predict(base_url, api_payload)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            st.error(f"API rejected the request ({error.code}): {detail}")
            return
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            st.error(f"Prediction failed: {error}")
            return
        predictions = pd.DataFrame(result["predictions"])
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
        chart = predictions.set_index("timestamp")[["y_pred"]].rename(
            columns={"y_pred": "Model forecast"}
        )
        chart["Persistence baseline"] = result["persistence_baseline"]
        if "future_actual" in payload:
            actual = payload["future_actual"]
            if len(actual) != len(chart):
                st.error("future_actual must contain exactly 72 values")
                return
            chart["Actual"] = actual
        st.subheader("72-hour forecast")
        st.line_chart(chart)
        left, right = st.columns(2)
        left.metric("Inference latency", f"{result['latency_ms']:.1f} ms")
        right.metric("Forecast unit", predictions["unit"].iloc[0])
        st.dataframe(predictions, use_container_width=True)
        if result.get("attention_weights") is not None:
            from src.viz.attention_heatmap import plot_attention_heatmap

            st.subheader("Attention heatmap")
            figure = plot_attention_heatmap(
                result["attention_weights"], run_id=result["model"]["run_id"],
                model_id=result["model"]["model_name"], data_split="inference",
                input_timestamps=payload["timestamps"],
                target_timestamps=predictions["timestamp"].astype(str).tolist(),
            )
            st.pyplot(figure)


if __name__ == "__main__":
    main()
