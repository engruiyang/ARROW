"""Scoring helpers for arrow candidates and no-candidate cases."""

from arrow_score import calibration as calibration_utils
from arrow_score import target_model
from arrow_score.types import ArrowCandidate, Calibration, ImagePair, ScoreResult, TargetSpec


class BasicScorer:
    """Convert arrow candidates into score results."""

    def score_candidate(
        self,
        candidate: ArrowCandidate,
        calibration: Calibration,
        target_spec: TargetSpec,
        pair: ImagePair,
        pair_index: int,
    ) -> ScoreResult:
        """Score one arrow candidate using calibrated target geometry."""
        radius_mm = calibration_utils.point_to_radius_mm(candidate.impact_point_px, calibration, target_spec)
        score = target_model.score_radius_mm(radius_mm, target_spec)
        needs_review = candidate.confidence == "low" or score == "outside"
        return ScoreResult(
            pair_index=pair_index,
            frame_prev=str(pair.prev.path),
            frame_curr=str(pair.curr.path),
            target_type=target_spec.name,
            impact_point_px=candidate.impact_point_px,
            radius_mm=radius_mm,
            score=score,
            confidence=candidate.confidence,
            needs_review=needs_review,
            debug_reason=candidate.debug_reason,
        )


def make_no_candidate_result(pair: ImagePair, pair_index: int, target_type: str, reason: str) -> ScoreResult:
    """Build a low-confidence result when no arrow candidate is detected."""
    return ScoreResult(
        pair_index=pair_index,
        frame_prev=str(pair.prev.path),
        frame_curr=str(pair.curr.path),
        target_type=target_type,
        impact_point_px=None,
        radius_mm=None,
        score=None,
        confidence="low",
        needs_review=True,
        debug_reason=reason,
    )
