import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fantasy_picker.models import (
    ModelArtifactError,
    load_model_artifacts,
)
from scripts.export_model_artifacts import (
    DEFAULT_SOURCE_NOTEBOOK,
    TARGET_TYPE,
    export_current_models,
)


POSITIONS = ("QB", "RB", "WR", "TE")


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, values):
        return [self.value] * len(values)


def notebook_objects():
    models = {
        position: ConstantModel(index)
        for index, position in enumerate(POSITIONS)
    }
    features = {position: ["feature_a"] for position in POSITIONS}
    medians = {position: {"feature_a": 1.5} for position in POSITIONS}
    return models, features, medians


class ExportModelArtifactTests(unittest.TestCase):
    def test_export_uses_existing_artifact_save_helper(self):
        models, features, medians = notebook_objects()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "models.pkl"
            with patch(
                "scripts.export_model_artifacts.save_model_artifacts"
            ) as save:
                artifacts = export_current_models(
                    final_models=models,
                    final_features=features,
                    final_medians=medians,
                    output_path=destination,
                )

        save.assert_called_once_with(artifacts, destination)

    def test_required_metadata_is_included(self):
        models, features, medians = notebook_objects()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = export_current_models(
                final_models=models,
                final_features=features,
                final_medians=medians,
                output_path=Path(directory) / "models.pkl",
                metadata={"notes": "test export"},
            )

        self.assertEqual(artifacts.metadata["artifact_version"], "1")
        self.assertEqual(artifacts.metadata["training_seasons"], [2021, 2022, 2023])
        self.assertEqual(artifacts.metadata["target_type"], TARGET_TYPE)
        self.assertEqual(artifacts.metadata["supported_positions"], list(POSITIONS))
        self.assertEqual(artifacts.metadata["source_notebook"], DEFAULT_SOURCE_NOTEBOOK)
        self.assertEqual(artifacts.metadata["notes"], "test export")
        datetime.fromisoformat(artifacts.metadata["creation_timestamp"])

    def test_missing_position_fails_clearly(self):
        models, features, medians = notebook_objects()
        del models["TE"]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ModelArtifactError, "TE"):
                export_current_models(
                    final_models=models,
                    final_features=features,
                    final_medians=medians,
                    output_path=Path(directory) / "models.pkl",
                )

    def test_exported_output_loads_through_package_loader(self):
        models, features, medians = notebook_objects()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "models.pkl"
            export_current_models(
                final_models=models,
                final_features=features,
                final_medians=medians,
                output_path=destination,
                training_seasons=(2022, 2023),
            )
            loaded = load_model_artifacts(destination)

        self.assertEqual(loaded.final_features, features)
        self.assertEqual(loaded.final_medians, medians)
        self.assertEqual(loaded.metadata["training_seasons"], [2022, 2023])
        self.assertEqual(loaded.final_models["QB"].predict([[0]])[0], 0)

    def test_exported_pickle_path_is_ignored_by_git(self):
        repository_root = Path(__file__).resolve().parents[1]
        ignored = subprocess.run(
            ["git", "check-ignore", "artifacts/fantasy_models.pkl"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(ignored.returncode, 0, ignored.stderr)
        self.assertEqual(ignored.stdout.strip(), "artifacts/fantasy_models.pkl")


if __name__ == "__main__":
    unittest.main()
