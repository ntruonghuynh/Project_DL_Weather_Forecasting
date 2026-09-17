"""Start FastAPI against one real ModelBundle and record a real /predict call.

Verifies the deployment chain end-to-end (bundle -> API -> Predictor -> model)
with a real 168h payload (scripts/generate_demo_payload.py output), and saves
the full request/response exchange to a JSON file for later inspection.
Never trains or fabricates a response; if the API returns an error, that
error is recorded as-is.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
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


def _start_server(bundle: Path, port: int) -> subprocess.Popen:
    environment = os.environ.copy()
    environment["JENA_MODEL_BUNDLE"] = str(bundle.resolve())
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _wait_for_health(
    server: subprocess.Popen,
    base_url: str,
    timeout: float,
) -> tuple[int, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            output = server.stdout.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"uvicorn exited before becoming ready:\n{output}")
        try:
            return _get(base_url, "/health")
        except (URLError, ConnectionError):
            time.sleep(0.25)
    raise TimeoutError(f"API did not answer within {timeout} seconds")


def _stop_server(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    server = _start_server(args.bundle, args.port)
    result: dict = {"bundle": str(args.bundle)}
    try:
        health_status, health_body = _wait_for_health(
            server, base_url, args.startup_timeout
        )
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
        invalid_requests = {}
        missing = copy.deepcopy(predict_request)
        missing["observations"][0].pop(next(iter(missing["observations"][0])))
        invalid_requests["missing_feature"] = _post(base_url, "/predict", missing)

        wrong_length = copy.deepcopy(predict_request)
        wrong_length["timestamps"] = wrong_length["timestamps"][:-1]
        wrong_length["observations"] = wrong_length["observations"][:-1]
        invalid_requests["wrong_sequence_length"] = _post(
            base_url, "/predict", wrong_length
        )

        not_finite = copy.deepcopy(predict_request)
        first_feature = next(iter(not_finite["observations"][0]))
        not_finite["observations"][0][first_feature] = float("nan")
        invalid_requests["nan"] = _post(base_url, "/predict", not_finite)

        wrong_order = copy.deepcopy(predict_request)
        wrong_order["observations"] = [
            dict(reversed(list(row.items()))) for row in wrong_order["observations"]
        ]
        invalid_requests["wrong_feature_order"] = _post(
            base_url, "/predict", wrong_order
        )
        result["invalid_requests"] = {
            name: {"status_code": status, "body": body}
            for name, (status, body) in invalid_requests.items()
        }
    finally:
        _stop_server(server)

    restart_server = _start_server(args.bundle, args.port)
    try:
        _wait_for_health(restart_server, base_url, args.startup_timeout)
        restart_status, restart_body = _post(base_url, "/predict", predict_request)
        original = [item["y_pred"] for item in predict_body.get("predictions", [])]
        restarted = [item["y_pred"] for item in restart_body.get("predictions", [])]
        max_difference = (
            max(abs(left - right) for left, right in zip(original, restarted, strict=True))
            if original and len(original) == len(restarted)
            else None
        )
        result["restart_determinism"] = {
            "status_code": restart_status,
            "prediction_count": len(restarted),
            "max_absolute_difference_degC": max_difference,
        }
    finally:
        _stop_server(restart_server)

    with tempfile.TemporaryDirectory(prefix="jena_corrupt_bundle_") as temporary:
        corrupt_bundle = Path(temporary) / "bundle"
        shutil.copytree(args.bundle, corrupt_bundle)
        (corrupt_bundle / "model.pt").write_bytes(b"corrupt")
        corrupt_url = f"http://127.0.0.1:{args.port + 1}"
        corrupt_server = _start_server(corrupt_bundle, args.port + 1)
        try:
            corrupt_status, corrupt_health = _wait_for_health(
                corrupt_server, corrupt_url, args.startup_timeout
            )
            model_info_status, model_info_body = _get(corrupt_url, "/model-info")
            result["corrupt_bundle"] = {
                "health": {"status_code": corrupt_status, "body": corrupt_health},
                "model_info": {
                    "status_code": model_info_status,
                    "body": model_info_body,
                },
            }
        finally:
            _stop_server(corrupt_server)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")
    print(f"/health -> {result['health']['status_code']}")
    print(f"/model-info -> {result['model_info']['status_code']}")
    print(f"/predict -> {result['predict']['status_code']}")
    print(
        "invalid requests -> "
        + ", ".join(
            f"{name}={value['status_code']}"
            for name, value in result["invalid_requests"].items()
        )
    )
    print(
        "restart max diff -> "
        f"{result['restart_determinism']['max_absolute_difference_degC']}"
    )
    print(f"corrupt bundle health -> {result['corrupt_bundle']['health']['body']['status']}")


if __name__ == "__main__":
    main()
