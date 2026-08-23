import pickle
import tempfile
import unittest
from pathlib import Path

from fantasy_picker.models import (
    ModelArtifactError,
    ModelArtifacts,
    load_model_artifacts,
    save_model_artifacts,
    validate_model_artifacts,
)


POSITIONS = ("QB", "RB", "WR", "TE")


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, values):
        return [self.value] * len(values)


def valid_artifacts():
    return ModelArtifacts(
        final_models={position: ConstantModel(index) for index, position in enumerate(POSITIONS)},
        final_features={position: ["feature_a", "feature_b"] for position in POSITIONS},
        final_medians={
            position: {"feature_a": 1.0, "feature_b": 2.0}
            for position in POSITIONS
        },
        metadata={"trained_through": 2025, "description": "test bundle"},
        version="2026.1",
    )


class ModelArtifactTests(unittest.TestCase):
    def test_valid_artifacts_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.pkl"
            save_model_artifacts(valid_artifacts(), path)
            loaded = load_model_artifacts(path)

        self.assertEqual(loaded.version, "2026.1")
        self.assertEqual(loaded.metadata["trained_through"], 2025)
        self.assertEqual(loaded.final_models["WR"].predict([[0]])[0], 2)

    def test_feature_names_and_medians_survive_round_trip(self):
        original = valid_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.pkl"
            save_model_artifacts(original, path)
            loaded = load_model_artifacts(path)

        self.assertEqual(loaded.final_features, original.final_features)
        self.assertEqual(loaded.final_medians, original.final_medians)

    def test_all_four_positions_are_required(self):
        artifacts = valid_artifacts()
        models = dict(artifacts.final_models)
        del models["TE"]
        malformed = ModelArtifacts(
            models,
            artifacts.final_features,
            artifacts.final_medians,
        )

        with self.assertRaisesRegex(ModelArtifactError, "TE"):
            validate_model_artifacts(malformed)

    def test_missing_feature_median_fails_clearly(self):
        artifacts = valid_artifacts()
        medians = {position: dict(values) for position, values in artifacts.final_medians.items()}
        del medians["RB"]["feature_b"]
        malformed = ModelArtifacts(
            artifacts.final_models,
            artifacts.final_features,
            medians,
        )

        with self.assertRaisesRegex(ModelArtifactError, "feature_b"):
            validate_model_artifacts(malformed)

    def test_malformed_saved_payload_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pkl"
            with path.open("wb") as artifact_file:
                pickle.dump({"final_models": {}}, artifact_file)

            with self.assertRaisesRegex(ModelArtifactError, "required fields"):
                load_model_artifacts(path)


if __name__ == "__main__":
    unittest.main()
