"""Lineup-level confidence helpers for fitted probability models."""

from typing import Optional, Protocol, Sequence


class ConfidenceModel(Protocol):
    """Minimal interface required from a fitted confidence model."""

    def predict_proba(
        self, values: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[float]]:
        ...


def get_lineup_confidence(
    lineup_gap: float, confidence_model: ConfidenceModel
) -> float:
    """Return the fitted model's positive-class probability for a lineup gap."""

    probabilities = confidence_model.predict_proba([[lineup_gap]])
    return float(probabilities[0][1])


def lineup_confidence_label(probability: Optional[float]) -> str:
    """Label lineup confidence using the notebook's final thresholds."""

    if probability is None:
        return "Unknown"
    if probability < 0.60:
        return "Low"
    if probability < 0.70:
        return "Moderate"
    if probability < 0.80:
        return "High"
    return "Very High"
