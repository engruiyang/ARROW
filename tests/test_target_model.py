import pytest

from arrow_score.target_model import get_target_spec, score_radius_mm


def test_get_target_spec_supports_string_and_int_90() -> None:
    assert get_target_spec("90").name == "90"
    assert get_target_spec(90).name == "90"


def test_get_target_spec_unsupported_type_raises() -> None:
    with pytest.raises(ValueError):
        get_target_spec("80")


@pytest.mark.parametrize(
    ("radius_mm", "expected"),
    [
        (0, 10),
        (45, 10),
        (45.1, 9),
        (95, 9),
        (95.1, 8),
        (145, 8),
        (145.1, 7),
        (195, 7),
        (195.1, 6),
        (245, 6),
        (245.1, "outside"),
    ],
)
def test_score_radius_mm_boundaries(radius_mm: float, expected: int | str) -> None:
    assert score_radius_mm(radius_mm, get_target_spec("90")) == expected


def test_score_radius_mm_negative_radius_raises() -> None:
    with pytest.raises(ValueError):
        score_radius_mm(-0.1, get_target_spec("90"))
