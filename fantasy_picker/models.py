"""Persistence helpers for trained fantasy-football model artifacts.

This module intentionally does not train models. Artifacts should only be loaded
from trusted sources because the pickle format can execute code while loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Mapping


REQUIRED_POSITIONS = ("QB", "RB", "WR", "TE")
ARTIFACT_VERSION = "1"


class ModelArtifactError(ValueError):
    """Raised when a model artifact is missing required inference data."""


@dataclass(frozen=True)
class ModelArtifacts:
    """The models and preprocessing values required for inference."""

    final_models: Mapping[str, Any]
    final_features: Mapping[str, list[str]]
    final_medians: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ARTIFACT_VERSION


def _require_position_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelArtifactError(f"{field_name} must be a mapping by position")

    missing = [position for position in REQUIRED_POSITIONS if position not in value]
    if missing:
        raise ModelArtifactError(
            f"{field_name} is missing required positions: {', '.join(missing)}"
        )
    return value


def _is_value_mapping(value: Any) -> bool:
    """Accept ordinary mappings and pandas Series-like median collections."""

    return isinstance(value, Mapping) or (
        callable(getattr(value, "keys", None))
        and hasattr(value, "__getitem__")
    )


def validate_model_artifacts(artifacts: ModelArtifacts) -> ModelArtifacts:
    """Validate and return a complete QB/RB/WR/TE artifact bundle."""

    if not isinstance(artifacts, ModelArtifacts):
        raise ModelArtifactError("artifacts must be a ModelArtifacts instance")

    models = _require_position_mapping(artifacts.final_models, "final_models")
    features = _require_position_mapping(artifacts.final_features, "final_features")
    medians = _require_position_mapping(artifacts.final_medians, "final_medians")

    for position in REQUIRED_POSITIONS:
        if not callable(getattr(models[position], "predict", None)):
            raise ModelArtifactError(
                f"final_models[{position!r}] must provide a callable predict method"
            )

        position_features = features[position]
        if (
            not isinstance(position_features, (list, tuple))
            or not position_features
            or not all(isinstance(name, str) and name for name in position_features)
        ):
            raise ModelArtifactError(
                f"final_features[{position!r}] must be a non-empty feature list"
            )

        position_medians = medians[position]
        if not _is_value_mapping(position_medians):
            raise ModelArtifactError(
                f"final_medians[{position!r}] must be a feature-to-value mapping"
            )
        missing_medians = [
            name for name in position_features if name not in position_medians
        ]
        if missing_medians:
            raise ModelArtifactError(
                f"final_medians[{position!r}] is missing features: "
                f"{', '.join(missing_medians)}"
            )

    if not isinstance(artifacts.metadata, Mapping):
        raise ModelArtifactError("metadata must be a mapping")
    if not isinstance(artifacts.version, str) or not artifacts.version:
        raise ModelArtifactError("version must be a non-empty string")

    return artifacts


def coerce_model_artifacts(value: ModelArtifacts | Mapping[str, Any]) -> ModelArtifacts:
    """Convert a serialized-style mapping to validated model artifacts."""

    if isinstance(value, ModelArtifacts):
        return validate_model_artifacts(value)
    if not isinstance(value, Mapping):
        raise ModelArtifactError("model artifacts must be a mapping or ModelArtifacts")

    required_fields = ("final_models", "final_features", "final_medians")
    missing = [name for name in required_fields if name not in value]
    if missing:
        raise ModelArtifactError(
            f"artifact is missing required fields: {', '.join(missing)}"
        )

    artifacts = ModelArtifacts(
        final_models=value["final_models"],
        final_features=value["final_features"],
        final_medians=value["final_medians"],
        metadata=value.get("metadata", {}),
        version=value.get("version", ARTIFACT_VERSION),
    )
    return validate_model_artifacts(artifacts)


def save_model_artifacts(
    artifacts: ModelArtifacts | Mapping[str, Any], path: str | Path
) -> None:
    """Validate and save a trusted local artifact bundle using pickle."""

    validated = coerce_model_artifacts(artifacts)
    payload = {
        "final_models": dict(validated.final_models),
        "final_features": {
            position: list(names)
            for position, names in validated.final_features.items()
        },
        "final_medians": {
            position: dict(values)
            for position, values in validated.final_medians.items()
        },
        "metadata": dict(validated.metadata),
        "version": validated.version,
    }

    try:
        with Path(path).open("wb") as artifact_file:
            pickle.dump(payload, artifact_file, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PickleError) as exc:
        raise ModelArtifactError(f"could not save model artifact: {exc}") from exc


def load_model_artifacts(path: str | Path) -> ModelArtifacts:
    """Load and validate a trusted local pickle artifact bundle."""

    try:
        with Path(path).open("rb") as artifact_file:
            payload = pickle.load(artifact_file)
    except (
        OSError,
        pickle.PickleError,
        EOFError,
        AttributeError,
        ImportError,
    ) as exc:
        raise ModelArtifactError(f"could not load model artifact: {exc}") from exc

    return coerce_model_artifacts(payload)
