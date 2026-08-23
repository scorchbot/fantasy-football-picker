"""Export the authoritative notebook's already-trained inference artifacts.

This helper does not train or load models. Run the notebook through the cell that
creates ``final_models``, ``final_features``, and ``final_medians``, then use it
from Colab while those objects are still in memory::

    from scripts.export_model_artifacts import export_current_models

    export_current_models(
        final_models=final_models,
        final_features=final_features,
        final_medians=final_medians,
        output_path="artifacts/fantasy_models.pkl",
        metadata={"notes": "Exported after the final model-training cell"},
    )

The default training seasons and source name reflect the current authoritative
``Fantasy_Football_Picker.ipynb`` notebook.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fantasy_picker.models import (
    ARTIFACT_VERSION,
    REQUIRED_POSITIONS,
    ModelArtifacts,
    save_model_artifacts,
)


DEFAULT_TRAINING_SEASONS = (2021, 2022, 2023)
DEFAULT_SOURCE_NOTEBOOK = "Fantasy_Football_Picker.ipynb"
TARGET_TYPE = "direct fantasy-points model"


def _validate_training_seasons(training_seasons: Sequence[int]) -> list[int]:
    seasons = list(training_seasons)
    if not seasons or any(
        isinstance(season, bool) or not isinstance(season, int)
        for season in seasons
    ):
        raise ValueError("training_seasons must contain one or more integers")
    return seasons


def export_current_models(
    *,
    final_models: Mapping[str, Any],
    final_features: Mapping[str, Sequence[str]],
    final_medians: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    training_seasons: Sequence[int] = DEFAULT_TRAINING_SEASONS,
    artifact_version: str = ARTIFACT_VERSION,
    source_notebook: str = DEFAULT_SOURCE_NOTEBOOK,
) -> ModelArtifacts:
    """Export prepared notebook objects through the package artifact saver."""

    supplied_metadata = dict(metadata or {})
    required_metadata = {
        "artifact_version": artifact_version,
        "training_seasons": _validate_training_seasons(training_seasons),
        "target_type": TARGET_TYPE,
        "supported_positions": list(REQUIRED_POSITIONS),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_notebook": source_notebook,
    }
    artifacts = ModelArtifacts(
        final_models=final_models,
        final_features=final_features,
        final_medians=final_medians,
        metadata={**supplied_metadata, **required_metadata},
        version=artifact_version,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_model_artifacts(artifacts, destination)
    return artifacts
