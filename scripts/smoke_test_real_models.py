"""Smoke-test an exported real-model artifact without training or data access.

Run from the repository root::

    python -m scripts.smoke_test_real_models
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from fantasy_picker.models import REQUIRED_POSITIONS, load_model_artifacts
from fantasy_picker.projections import project_fantasy_points


DEFAULT_ARTIFACT_PATH = Path("artifacts/fantasy_models.pkl")


def _median_feature_row(
    feature_names: list[str] | tuple[str, ...],
    medians: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a synthetic inference row using every stored feature median."""

    return {feature_name: medians[feature_name] for feature_name in feature_names}


def smoke_test_real_models(
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, float]:
    """Load an artifact and return one finite median-row projection per position."""

    artifacts = load_model_artifacts(artifact_path)
    projections = {}

    for position in REQUIRED_POSITIONS:
        feature_row = _median_feature_row(
            artifacts.final_features[position],
            artifacts.final_medians[position],
        )
        projection = project_fantasy_points(
            feature_row,
            position,
            artifacts.final_features,
            artifacts.final_medians,
            artifacts.final_models,
        )
        if not math.isfinite(projection):
            raise RuntimeError(
                f"{position} projection is not a finite numeric value: {projection!r}"
            )
        projections[position] = projection

    return projections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run median-feature inference against exported fantasy models."
    )
    parser.add_argument(
        "artifact_path",
        nargs="?",
        default=DEFAULT_ARTIFACT_PATH,
        help="Artifact path (default: artifacts/fantasy_models.pkl)",
    )
    arguments = parser.parse_args(argv)

    try:
        projections = smoke_test_real_models(arguments.artifact_path)
    except Exception as exc:
        print(f"FAILURE: real-model smoke test failed: {exc}", file=sys.stderr)
        return 1

    for position in REQUIRED_POSITIONS:
        print(f"{position} projection: {projections[position]:.4f}")
    print("SUCCESS: all four real-model projections are finite numeric values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
