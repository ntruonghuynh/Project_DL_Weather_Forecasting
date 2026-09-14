"""Attention heatmap plotting + demo-data export (owner: TV4).

Explanation/debug tool only - per the mission spec, this must NEVER be
used to decide model selection; the shared evaluation metric does that.

Compliant with agents/rules/VISUALIZATION_RULES.md:
  - every figure carries a title, axis labels WITH UNITS, a colorbar
    (acts as the "legend" for a single continuous-value heatmap), and the
    data split + run/model identifier it came from
  - figures saved into report/figures/ get an entry appended to
    report/figure_manifest.csv (path, run_id, split, description) via
    `record_figure_in_manifest`, so they stay traceable to a run
  - callers of `save_report_figure` are required to supply a
    Finding -> Impact -> Decision caption; "Decision" must say what
    evidence is still needed if there isn't one yet - it is never invented

Suggested location in the repo: src/viz/attention_heatmap.py (adjust
import path below if the team places viz utilities elsewhere, e.g.
src/interpretability/ or scripts/).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FIGURE_MANIFEST_COLUMNS = [
    "path",
    "run_id",
    "model_id",
    "split",
    "description",
    "finding",
    "impact",
    "decision",
]


def plot_attention_heatmap(
    attention_weights,
    run_id: str,
    model_id: str,
    data_split: str,
    sample_index: int | None = None,
    input_timestamps=None,
    target_timestamps=None,
    ax=None,
):
    """Plot a single sample's attention heatmap: decoder step x encoder time.

    Parameters
    ----------
    attention_weights : array-like, shape [horizon, T_enc]
        Attention weights for ONE sample (already selected/sliced out of a
        batch - this function does not index into a batch dimension).
    run_id : str
        Immutable run identifier this figure was generated from (required
        by VISUALIZATION_RULES.md - "experiment figures MUST record
        run/model identifier"). Use the real run registry ID, not a
        made-up label.
    model_id : str
        Model/config identifier (e.g. "attention_lstm_seq2seq").
    data_split : str
        Which split these attention weights came from, e.g. "validation".
        Required by VISUALIZATION_RULES.md ("figure MUST ghi data split").
        NEVER pass "test" here for anything other than a final, locked
        evaluation per EXPERIMENT_RULES.md / EVALUATION_RULES.md.
    sample_index : optional int
        Index of this sample within its batch/split, for traceability.
    input_timestamps, target_timestamps : optional array-like
        Real timestamps for axis annotation. Cosmetic only, but strongly
        recommended over bare step indices when available.
    ax : optional matplotlib Axes
        If provided, draw onto this axes instead of creating a new figure.

    Returns
    -------
    matplotlib Figure (the one `ax` belongs to, or a newly created one).
    """
    import matplotlib.pyplot as plt

    weights = np.asarray(attention_weights)
    if weights.ndim != 2:
        raise ValueError(
            f"attention_weights must be 2D [horizon, T_enc], got shape {weights.shape}"
        )
    if not run_id or not model_id or not data_split:
        raise ValueError(
            "run_id, model_id and data_split are required (VISUALIZATION_RULES.md: "
            "every experiment figure must record run/model identifier and data split)"
        )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(weights, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Encoder time step (hours before forecast start)")
    ax.set_ylabel("Decoder step (forecast hour ahead, 1-72)")
    fig.colorbar(
        im,
        ax=ax,
        label="Attention weight (softmax over encoder time, unitless, sums to 1 per row)",
    )

    sample_part = f" | sample={sample_index}" if sample_index is not None else ""
    title = f"Attention weights — model={model_id} run={run_id} split={data_split}{sample_part}"
    ax.set_title(title, fontsize=10)

    if input_timestamps is not None:
        ax.set_xlabel(
            f"Encoder time step (hours before forecast start)\n"
            f"input range: {input_timestamps[0]} to {input_timestamps[-1]}"
        )
    if target_timestamps is not None:
        ax.set_ylabel(
            f"Decoder step (forecast hour ahead, 1-72)\n"
            f"target range: {target_timestamps[0]} to {target_timestamps[-1]}"
        )

    if created_fig:
        fig.tight_layout()

    return fig


def save_report_figure(
    fig,
    filename: str,
    run_id: str,
    model_id: str,
    data_split: str,
    description: str,
    finding: str,
    impact: str,
    decision: str,
    report_dir: str | Path = "report",
) -> Path:
    """Save a figure into report/figures/ and register it in figure_manifest.csv.

    Required by VISUALIZATION_RULES.md:
      - "Figure đưa vào report MUST được xuất từ code có thể chạy lại" - this
        function itself IS that reproducible export step; don't hand-edit
        images or manifest rows afterward.
      - "Figure report MUST có entry trong report/figure_manifest.csv"
      - "File figure MUST nằm trong report/figures/"
      - caption MUST state Finding -> Impact -> Decision; if there is no
        decision yet, `decision` must say what evidence is still needed -
        never invent a conclusion.

    Do NOT call this for exploratory/debug-only heatmaps you are not
    actually including in the report - only for figures that go in
    report/figures/ and get cited there.
    """
    if not finding or not impact or not decision:
        raise ValueError(
            "finding, impact and decision are all required (VISUALIZATION_RULES.md: "
            "caption must state Finding -> Impact -> Decision; if there's no decision "
            "yet, 'decision' must say what evidence is still needed)"
        )

    report_dir = Path(report_dir)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figures_dir / filename
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")

    manifest_path = report_dir / "figure_manifest.csv"
    is_new = not manifest_path.exists()
    rel_path = (
        figure_path.relative_to(report_dir.parent)
        if report_dir.parent != Path(".")
        else figure_path
    )
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIGURE_MANIFEST_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "path": str(rel_path),
                "run_id": run_id,
                "model_id": model_id,
                "split": data_split,
                "description": description,
                "finding": finding,
                "impact": impact,
                "decision": decision,
            }
        )

    return figure_path


def export_attention_demo_data(
    attention_weights,
    predictions,
    run_id: str,
    model_id: str,
    data_split: str,
    targets=None,
    sample_indices=None,
    output_path: str | Path = "artifacts/attention_demo.json",
) -> Path:
    """Export a small JSON payload for the demo app to render without recomputing.

    Parameters
    ----------
    attention_weights : array-like [B, horizon, T_enc]
    predictions : array-like [B, horizon, 1] or [B, horizon]
    run_id, model_id, data_split : str
        Provenance metadata carried alongside the demo data so the demo
        app (and anyone auditing it later) can trace it back to a real
        run, per the same traceability principle as figure_manifest.csv.
    targets : optional array-like, same shape as predictions
    sample_indices : optional list[int]
        Which batch indices to export. Defaults to all of them - callers
        should pass a small, curated list (per the "sample selection"
        deliverable) rather than exporting a full validation batch.
    """
    weights = np.asarray(attention_weights)
    preds = np.asarray(predictions).reshape(weights.shape[0], weights.shape[1])
    targs = None if targets is None else np.asarray(targets).reshape(preds.shape)

    if sample_indices is None:
        sample_indices = list(range(weights.shape[0]))

    payload = {
        "run_id": run_id,
        "model_id": model_id,
        "split": data_split,
        "samples": [],
    }
    for idx in sample_indices:
        sample = {
            "sample_index": int(idx),
            "attention_weights": weights[idx].tolist(),
            "predictions": preds[idx].tolist(),
        }
        if targs is not None:
            sample["targets"] = targs[idx].tolist()
        payload["samples"].append(sample)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def select_interesting_samples(attention_weights, num_samples: int = 4) -> list[int]:
    """Pick a small, diverse set of samples for the demo heatmap gallery.

    Heuristic only (explanation/debug tool, not used for model selection):
    picks samples whose attention distribution entropy spans the observed
    range, so the demo shows both "sharp" and "diffuse" attention examples
    rather than num_samples near-identical ones.
    """
    weights = np.asarray(attention_weights)
    if weights.shape[0] <= num_samples:
        return list(range(weights.shape[0]))

    eps = 1e-12
    mean_weights = weights.mean(axis=1)  # average over decoder steps -> [B, T_enc]
    entropy = -(mean_weights * np.log(mean_weights + eps)).sum(axis=-1)  # [B]

    order = np.argsort(entropy)
    picks = np.linspace(0, len(order) - 1, num=num_samples, dtype=int)
    return sorted(int(order[i]) for i in picks)
