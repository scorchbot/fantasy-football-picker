import types
import unittest
from unittest.mock import MagicMock, patch

from fantasy_picker.data import nflverse


class FakePandasFrame:
    def __init__(self, columns, rows=None):
        self.columns = columns
        self.rows = rows or []

    @property
    def empty(self):
        return not self.rows


class FakePolarsFrame:
    def __init__(self, pandas_frame):
        self.pandas_frame = pandas_frame
        self.to_pandas_calls = 0

    def to_pandas(self):
        self.to_pandas_calls += 1
        return self.pandas_frame


PUBLIC_LOADERS = {
    "load_schedules": "load_schedules",
    "load_player_stats": "load_player_stats",
    "load_weekly_rosters": "load_rosters_weekly",
    "load_injuries": "load_injuries",
    "load_snap_counts": "load_snap_counts",
    "load_depth_charts": "load_depth_charts",
    "load_ff_opportunity": "load_ff_opportunity",
}


def fake_nflreadpy(result=None):
    module = types.SimpleNamespace()
    for nflreadpy_name in PUBLIC_LOADERS.values():
        setattr(module, nflreadpy_name, MagicMock(return_value=result))
    return module


class NFLVerseDataTests(unittest.TestCase):
    def test_each_public_loader_calls_nflreadpy_and_converts_to_pandas(self):
        pandas_frame = FakePandasFrame(
            columns=["season", "source_column"], rows=[[2025, "unchanged"]]
        )

        for public_name, nflreadpy_name in PUBLIC_LOADERS.items():
            with self.subTest(loader=public_name):
                polars_frame = FakePolarsFrame(pandas_frame)
                source = fake_nflreadpy(polars_frame)
                with patch.object(nflverse, "_get_nflreadpy", return_value=source):
                    result = getattr(nflverse, public_name)([2024, 2025])

                self.assertIs(result, pandas_frame)
                self.assertEqual(result.columns, ["season", "source_column"])
                self.assertEqual(polars_frame.to_pandas_calls, 1)
                getattr(source, nflreadpy_name).assert_called_once_with([2024, 2025])

    def test_integer_season_is_passed_without_hidden_state(self):
        polars_frame = FakePolarsFrame(FakePandasFrame(["season"], [[2025]]))
        source = fake_nflreadpy(polars_frame)

        with patch.object(nflverse, "_get_nflreadpy", return_value=source):
            nflverse.load_player_stats(2025)

        source.load_player_stats.assert_called_once_with(2025)

    def test_empty_results_remain_empty_pandas_frames(self):
        pandas_frame = FakePandasFrame(columns=["season"])
        source = fake_nflreadpy(FakePolarsFrame(pandas_frame))

        with patch.object(nflverse, "_get_nflreadpy", return_value=source):
            result = nflverse.load_injuries(2025)

        self.assertIs(result, pandas_frame)
        self.assertTrue(result.empty)

    def test_invalid_seasons_fail_before_loading(self):
        for seasons in (True, "2025", [], [2024, "2025"]):
            with self.subTest(seasons=seasons):
                with self.assertRaisesRegex(ValueError, "integer"):
                    nflverse.load_schedules(seasons)

    def test_missing_nflreadpy_has_clear_error(self):
        with patch.object(
            nflverse, "import_module", side_effect=ModuleNotFoundError("nflreadpy")
        ):
            with self.assertRaisesRegex(
                nflverse.NFLVerseDataError, "nflreadpy is required"
            ):
                nflverse.load_schedules(2025)

    def test_nflreadpy_load_errors_have_dataset_context(self):
        source = fake_nflreadpy()
        source.load_depth_charts.side_effect = OSError("download failed")

        with patch.object(nflverse, "_get_nflreadpy", return_value=source):
            with self.assertRaisesRegex(
                nflverse.NFLVerseDataError,
                r"nflreadpy\.load_depth_charts failed.*download failed",
            ):
                nflverse.load_depth_charts([2024, 2025])

    def test_current_week_inputs_load_every_raw_dataset_for_explicit_season(self):
        expected_keys = list(PUBLIC_LOADERS)
        patches = {
            name: patch.object(nflverse, name, return_value=f"{name}-frame")
            for name in expected_keys
        }

        mocks = {name: patcher.start() for name, patcher in patches.items()}
        self.addCleanup(lambda: [patcher.stop() for patcher in patches.values()])

        result = nflverse.load_current_week_inputs(2025)

        self.assertEqual(
            set(result),
            {
                "schedules",
                "player_stats",
                "weekly_rosters",
                "injuries",
                "snap_counts",
                "depth_charts",
                "ff_opportunity",
            },
        )
        for loader in mocks.values():
            loader.assert_called_once_with(2025)

    def test_current_week_inputs_requires_integer_season(self):
        with self.assertRaisesRegex(ValueError, "season must be an integer"):
            nflverse.load_current_week_inputs([2025])


if __name__ == "__main__":
    unittest.main()
