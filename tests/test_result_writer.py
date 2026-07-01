import csv
import json

from arrow_score.result_writer import JsonCsvResultWriter
from arrow_score.types import ScoreResult


def test_json_csv_result_writer_writes_files(tmp_path) -> None:
    result = ScoreResult(
        pair_index=0,
        frame_prev="prev.png",
        frame_curr="curr.png",
        target_type="90",
        impact_point_px=(100.0, 101.0),
        radius_mm=12.5,
        score=10,
        confidence="medium",
        needs_review=False,
    )

    JsonCsvResultWriter().write([result], tmp_path)

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    assert json_path.exists()
    assert csv_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["score"] == 10
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0][:4] == ["pair_index", "frame_prev", "frame_curr", "target_type"]
