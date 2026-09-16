"""Run FastAPI and Streamlit together against one versioned bundle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.bundle import verify_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--app-port", type=int, default=8501)
    args = parser.parse_args()
    verify_bundle(args.bundle)
    environment = os.environ.copy()
    environment["JENA_MODEL_BUNDLE"] = str(args.bundle.resolve())
    environment["JENA_API_URL"] = f"http://127.0.0.1:{args.api_port}"
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(args.api_port)],
        env=environment,
    )
    app = None
    try:
        time.sleep(2)
        if api.poll() is not None:
            raise RuntimeError("FastAPI failed to start")
        app = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
             "--server.port", str(args.app_port)],
            env=environment,
        )
        print(f"FastAPI: http://127.0.0.1:{args.api_port}")
        print(f"Streamlit: http://127.0.0.1:{args.app_port}")
        exit_code = app.wait()
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        pass
    finally:
        if app is not None and app.poll() is None:
            app.terminate()
        if api.poll() is None:
            api.terminate()


if __name__ == "__main__":
    main()
