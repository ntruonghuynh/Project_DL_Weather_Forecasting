import numpy as np

from src.evaluation.error_analysis import analyze_errors, build_error_frame


def test_error_analysis_returns_worst_and_grouped_metrics() -> None:
    timestamps = np.array(
        [["2026-01-01T01:00:00", "2026-01-01T02:00:00"],
         ["2026-07-01T01:00:00", "2026-07-01T02:00:00"]]
    )
    frame = build_error_frame(
        [[0, 0], [0, 0]], [[1, 4], [2, 3]], target_timestamps=timestamps,
        run_id="run", model_name="model", split="validation",
    )
    result = analyze_errors(frame, top_n=2)
    assert result["worst_cases"]["absolute_error"].tolist() == [4, 3]
    assert set(result) == {"worst_cases", "by_hour", "by_season", "by_horizon"}
    assert result["by_horizon"]["sample_count"].sum() == 4
