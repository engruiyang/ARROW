"""JSON and CSV result writing for the demo pipeline."""

import csv
import json
from pathlib import Path
from typing import Any

from arrow_score.types import ScoreResult

CSV_FIELDS = [
    "pair_index",
    "frame_prev",
    "frame_curr",
    "target_type",
    "x_px",
    "y_px",
    "radius_mm",
    "score",
    "confidence",
    "needs_review",
    "debug_reason",
]


def result_to_dict(result: ScoreResult) -> dict[str, Any]:
    """Convert a score result into JSON/CSV friendly values."""
    x_px = result.impact_point_px[0] if result.impact_point_px is not None else None
    y_px = result.impact_point_px[1] if result.impact_point_px is not None else None
    return {
        "pair_index": result.pair_index,
        "frame_prev": result.frame_prev,
        "frame_curr": result.frame_curr,
        "target_type": result.target_type,
        "impact_point_px": list(result.impact_point_px) if result.impact_point_px is not None else None,
        "x_px": x_px,
        "y_px": y_px,
        "radius_mm": result.radius_mm,
        "score": result.score,
        "confidence": result.confidence,
        "needs_review": result.needs_review,
        "debug_reason": result.debug_reason,
    }


class JsonCsvResultWriter:
    """Write score results to results.json and results.csv."""

    def write(self, results: list[ScoreResult], output_dir: Path) -> None:
        """Persist result files under output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [result_to_dict(result) for result in results]

        with (output_dir / "results.json").open("w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
            file.write("\n")

        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in CSV_FIELDS})
