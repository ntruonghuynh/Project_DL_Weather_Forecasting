"""Build a checksummed release bundle from one immutable trained run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.serving.bundle import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--model-kwargs", required=True, help="JSON object")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    kwargs = json.loads(args.model_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--model-kwargs must decode to a JSON object")
    result = build_bundle(
        args.destination, checkpoint=args.checkpoint, scaler=args.scaler,
        feature_schema=args.feature_schema, resolved_config=args.resolved_config,
        metadata=args.metadata, model_class=args.model_class, model_kwargs=kwargs,
        model_name=args.model_name, model_version=args.model_version, run_id=args.run_id,
    )
    print(result)
