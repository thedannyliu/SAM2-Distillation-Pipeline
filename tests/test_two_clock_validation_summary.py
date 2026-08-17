from __future__ import annotations

import json
from pathlib import Path

from tools.eval.summarize_two_clock_validation import collect


def _write_output(root: Path, *, jf: float, model_ms: float, wall_ms: float) -> None:
    root.mkdir(parents=True)
    (root / "pred").mkdir()
    (root / "sav_eval.json").write_text(
        json.dumps({"metrics": {"J&F": jf, "J": jf - 1, "F": jf + 1}}),
        encoding="utf-8",
    )
    (root / "pred/summary.json").write_text(
        json.dumps(
            {
                "two_clock_refresh_rate": 0.25,
                "single_stream_model_mean_ms": model_ms,
                "single_stream_wall_ms_per_frame": wall_ms,
                "rank_model_p95_ms": [model_ms + 2, model_ms + 3],
            }
        ),
        encoding="utf-8",
    )


def test_summary_compares_jf_and_latency_to_official_baselines(
    tmp_path: Path,
) -> None:
    _write_output(tmp_path / "controls/O0/R1", jf=80, model_ms=20, wall_ms=30)
    _write_output(tmp_path / "controls/O1/R4", jf=60, model_ms=10, wall_ms=15)
    rows, missing = collect(tmp_path)
    assert missing
    o0 = next(row for row in rows if row["model"] == "O0")
    o1 = next(row for row in rows if row["model"] == "O1")
    assert o0["delta_J&F_vs_O0_R1"] == 0
    assert o1["delta_J&F_vs_O0_R1"] == -20
    assert o1["delta_J&F_vs_official_matched"] == 0
    assert o1["model_speedup_vs_O0_R1"] == 2
    assert o1["wall_speedup_vs_O0_R1"] == 2
