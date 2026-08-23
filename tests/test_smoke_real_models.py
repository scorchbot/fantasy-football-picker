import io
import math
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from fantasy_picker.models import ModelArtifacts
from scripts.smoke_test_real_models import main, smoke_test_real_models


POSITIONS = ("QB", "RB", "WR", "TE")


class RecordingModel:
    def __init__(self, projection):
        self.projection = projection
        self.inputs = []

    def predict(self, values):
        if hasattr(values, "iloc"):
            self.inputs.append(values.iloc[0].to_dict())
        else:
            self.inputs.append(dict(zip(("feature_a", "feature_b"), values[0])))
        return [self.projection]


def fake_artifacts(projections=None):
    values = projections or {
        position: float(index + 10) for index, position in enumerate(POSITIONS)
    }
    models = {
        position: RecordingModel(values[position]) for position in POSITIONS
    }
    return ModelArtifacts(
        final_models=models,
        final_features={
            position: ["feature_a", "feature_b"] for position in POSITIONS
        },
        final_medians={
            position: {"feature_a": 1.25, "feature_b": 2.5}
            for position in POSITIONS
        },
    )


class RealModelSmokeTestTests(unittest.TestCase):
    def test_each_position_uses_its_exact_feature_medians(self):
        artifacts = fake_artifacts()
        with patch(
            "scripts.smoke_test_real_models.load_model_artifacts",
            return_value=artifacts,
        ) as load:
            projections = smoke_test_real_models("unused-test-artifact.pkl")

        load.assert_called_once_with("unused-test-artifact.pkl")
        self.assertEqual(set(projections), set(POSITIONS))
        for position in POSITIONS:
            with self.subTest(position=position):
                self.assertEqual(
                    artifacts.final_models[position].inputs,
                    [{"feature_a": 1.25, "feature_b": 2.5}],
                )
                self.assertTrue(math.isfinite(projections[position]))

    def test_non_finite_projection_fails_clearly(self):
        artifacts = fake_artifacts({
            "QB": 20.0,
            "RB": 15.0,
            "WR": float("inf"),
            "TE": 10.0,
        })
        with patch(
            "scripts.smoke_test_real_models.load_model_artifacts",
            return_value=artifacts,
        ):
            with self.assertRaisesRegex(RuntimeError, "WR projection"):
                smoke_test_real_models("unused-test-artifact.pkl")

    def test_command_reports_success_and_all_four_projections(self):
        with patch(
            "scripts.smoke_test_real_models.load_model_artifacts",
            return_value=fake_artifacts(),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["unused-test-artifact.pkl"])

        self.assertEqual(exit_code, 0)
        for position in POSITIONS:
            self.assertIn(f"{position} projection:", output.getvalue())
        self.assertIn("SUCCESS", output.getvalue())

    def test_command_reports_failure_with_nonzero_exit(self):
        with patch(
            "scripts.smoke_test_real_models.load_model_artifacts",
            side_effect=FileNotFoundError("missing artifact"),
        ):
            error = io.StringIO()
            with redirect_stderr(error):
                exit_code = main(["missing.pkl"])

        self.assertEqual(exit_code, 1)
        self.assertIn("FAILURE", error.getvalue())
        self.assertIn("missing artifact", error.getvalue())


if __name__ == "__main__":
    unittest.main()
