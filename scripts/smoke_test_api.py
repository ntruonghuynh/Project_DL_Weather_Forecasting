"""Start FastAPI against one real ModelBundle and record a real /predict call.

Verifies the deployment chain end-to-end (bundle -> API -> Predictor -> model)
with a real 168h payload (scripts/generate_demo_payload.py output), and saves
the full request/response exchange to a JSON file for later inspection.
Never trains or fabricates a response; if the API returns an error, that
error is recorded as-is.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get(base_url: str, path: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urlopen(f"{base_url}{path}", timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _post(base_url: str, path: str, payload: dict, timeout: float = 60.0) -> tuple[int, dict]:
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["JENA_MODEL_BUNDLE"] = str(args.bundle.resolve())
    base_url = f"http://127.0.0.1:{args.port}"

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(args.port)],
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    result: dict = {"bundle": str(args.bundle)}
    try:
        deadline = time.monotonic() + args.startup_timeout
        health_status, health_body = None, None
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("uvicorn exited before becoming ready")
            try:
                health_status, health_body = _get(base_url, "/health")
                break
            except (URLError, ConnectionError):
                time.sleep(0.5)
        result["health"] = {"status_code": health_status, "body": health_body}

        model_info_status, model_info_body = _get(base_url, "/model-info")
        result["model_info"] = {"status_code": model_info_status, "body": model_info_body}

        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        predict_request = {
            "timestamps": payload["timestamps"],
            "observations": payload["observations"],
        }
        started = time.monotonic()
        predict_status, predict_body = _post(base_url, "/predict", predict_request)
        result["predict"] = {
            "status_code": predict_status,
            "request_shape": {
                "timestamps": len(predict_request["timestamps"]),
                "observations": len(predict_request["observations"]),
            },
            "wall_clock_seconds": round(time.monotonic() - started, 4),
            "body": predict_body,
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")
    print(f"/health -> {result['health']['status_code']}")
    print(f"/model-info -> {result['model_info']['status_code']}")
    print(f"/predict -> {result['predict']['status_code']}")


if __name__ == "__main__":
    main()
