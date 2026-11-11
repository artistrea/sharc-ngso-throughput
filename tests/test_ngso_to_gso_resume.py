"""
Regression tests for resuming interrupted ngso_to_gso runs.

A resumed run must produce exactly the same results as an uninterrupted
one. Bugs this guards against: resume never being offered (results.json
not written, results looked up as csv instead of parquet, timestamp parsed
from the wrong directory), resuming from drop 0, overwriting the rows of
the previous run, unreadable parquet files after an interrupt, and RAND
satellite selection depending on where the run started.
"""
import itertools
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

import sharc.ngso_to_gso as ngso_to_gso
from sharc.ngso_to_gso import ResultsWriter, run_simulation
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR,
)

N_STEPS = 60
BATCH_SIZE = 10


def make_param_file(tmp_dir: Path, scenario_name: str) -> Path:
    """Write a small RAND scenario with two links at two stations."""
    link = {
        "label": "DirecTV 15 GW", "orbital_slot_deg": -102.75,
        "center_freq_GHz": 19.95, "bandwidth_MHz": 36.0,
        "eirp_dBW_per_carrier": 59.5, "peak_rx_antenna_gain": 65.4,
        "rx_antenna_size_m": 13.2, "g_over_t_dB_per_K": 41.4,
        "station_type": "GW",
    }
    stations = [
        {"label": "Las Vegas", "lat_deg": 36.19, "lon_deg": -115.18},
        {"label": "Kansas", "lat_deg": 39.07, "lon_deg": -94.6},
    ]
    raw = {"ngso2gso": {
        "version": 2,
        "scenario_name": scenario_name,
        "seed": 11,
        "delta_t_s": 10.0,
        "min_t_s": 0.0,
        "max_t_s": 10.0 * N_STEPS,
        "batch_size": BATCH_SIZE,
        "minimum_elevation": 20.0,
        "selection_strategy": "RAND",
        "ngso_systems": [{
            "label": "NGSO",
            "n_co_channel": 4,
            "gso_protection_avoidance_angle": 5.0,
            "tx_model": {
                "model": "CONSTANT_PFD_AT_GND",
                "pfd_at_ref_bandwidth_dBW_m2": -115.0,
            },
            "orbits": [{
                "n_planes": 36, "sats_per_plane": 22,
                "inclination_deg": 53.0, "phasing_deg": 3.3,
                "perigee_alt_km": 550.0, "apogee_alt_km": 550.0,
            }],
        }],
        "gso_links": [{**link, "earth_station": es} for es in stations],
    }}
    path = tmp_dir / f"{scenario_name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def load_params(param_file: Path) -> ParametersNGSO2GSO:
    """Load and validate a parameter file."""
    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(param_file)
    par.validate("ngso2gso")
    return par


class FakeClock(datetime):
    """datetime whose now() advances one second per call.

    Results directories are named after the current second, and these
    runs take less than a second, so each run needs its own timestamp.
    """

    _ticks = itertools.count()

    @classmethod
    def now(cls, tz=None):
        """Return a new, later time on each call."""
        return datetime(2026, 1, 1) + timedelta(seconds=next(cls._ticks))


class MemoryResultsWriter:
    """Stand-in for ResultsWriter that keeps results in memory."""

    def __init__(self):
        """Start with no results."""
        self.rows = {}

    def add_results(self, new_res: dict, attr_name: str):
        """Store one row of results."""
        self.rows.setdefault(attr_name, []).append(dict(new_res))

    def flush_results(self):
        """Do nothing, results are kept in memory."""

    def close(self, *args, **kwargs):
        """Do nothing, results are kept in memory."""


class NGSO2GSOResumeTest(unittest.TestCase):
    """Interrupting and resuming a run must not change its results."""

    def setUp(self):
        """Send results to a temporary directory and fake the clock."""
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        results_dir = self.tmp / "results"
        results_dir.mkdir()
        for target, value in (
            ("RESULTS_DIR", str(results_dir)),
            ("RESULTS_JSON_RESUME_PATH", results_dir / "results.json"),
            ("datetime", FakeClock),
        ):
            patcher = mock.patch.object(ngso_to_gso, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def run_main(self, param_file: Path, answer="Y", interrupt_at_flush=None):
        """
        Run ngso_to_gso.main() on param_file.

        answer is given to the "continue previous simulation?" prompt.
        If interrupt_at_flush is set, a KeyboardInterrupt is raised when
        that results flush starts (1 = first chunk), like Ctrl-C would.
        Returns the input() mock, to check whether resume was offered.
        Sets self.rows_computed to the number of result rows the run
        computed (time steps x links).
        """
        original_flush = ResultsWriter.flush_results
        original_add = ResultsWriter.add_results
        n_flushes = itertools.count(1)
        self.rows_computed = 0

        def add_results(writer, *args, **kwargs):
            self.rows_computed += 1
            original_add(writer, *args, **kwargs)

        def flush(writer):
            if next(n_flushes) == interrupt_at_flush:
                raise KeyboardInterrupt
            original_flush(writer)

        argv = ["sharc", "-p", str(param_file)]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("builtins.input", return_value=answer) as inp, \
                mock.patch.object(ResultsWriter, "flush_results", flush), \
                mock.patch.object(ResultsWriter, "add_results", add_results):
            if interrupt_at_flush is None:
                ngso_to_gso.main()
            else:
                with self.assertRaises(KeyboardInterrupt):
                    ngso_to_gso.main()
        return inp

    def latest_dir(self, scenario_name: str) -> Path:
        """Results directory results.json points to for a scenario."""
        resume = json.loads(
            ngso_to_gso.RESULTS_JSON_RESUME_PATH.read_text()
        )
        return Path(resume[scenario_name]["latest_res_dir"])

    def assert_same_results(self, got: Path, expected: Path):
        """Check two results directories hold identical parquet files."""
        files = sorted(p.name for p in expected.glob("*.parquet"))
        self.assertEqual(len(files), 2)
        self.assertEqual(sorted(p.name for p in got.glob("*.parquet")), files)
        self.assertEqual(list(got.glob("*.tmp")), [])
        for name in files:
            exp = pd.read_parquet(expected / name)
            self.assertEqual(len(exp), N_STEPS)
            self.assertTrue(np.isfinite(exp["i"]).all())
            pd.testing.assert_frame_equal(
                pd.read_parquet(got / name), exp, check_exact=True,
                obj=name,
            )

    def reference_run(self) -> Path:
        """Uninterrupted run of the scenario, for comparison."""
        self.run_main(make_param_file(self.tmp, "reference"))
        return self.latest_dir("reference")

    def test_interrupted_then_resumed_matches_uninterrupted(self):
        """Resume is offered and gives the uninterrupted results."""
        expected = self.reference_run()
        param_file = make_param_file(self.tmp, "resumed")

        self.run_main(param_file, interrupt_at_flush=3)
        interrupted = self.latest_dir("resumed")
        inp = self.run_main(param_file, answer="Y")

        inp.assert_called_once()
        # only the drops that were not finished are run again
        self.assertEqual(self.rows_computed, 2 * (N_STEPS - 2 * BATCH_SIZE))
        self.assertEqual(self.latest_dir("resumed"), interrupted)
        self.assert_same_results(interrupted, expected)

    def test_interrupted_twice_then_resumed_matches_uninterrupted(self):
        """A resumed run can itself be interrupted and resumed."""
        expected = self.reference_run()
        param_file = make_param_file(self.tmp, "resumed")

        self.run_main(param_file, interrupt_at_flush=2)
        self.run_main(param_file, answer="Y", interrupt_at_flush=3)
        self.run_main(param_file, answer="Y")

        self.assert_same_results(self.latest_dir("resumed"), expected)

    def test_interrupted_run_keeps_complete_chunks_readable(self):
        """Interrupt leaves readable files holding only finished chunks."""
        self.run_main(
            make_param_file(self.tmp, "resumed"), interrupt_at_flush=4,
        )
        res_dir = self.latest_dir("resumed")
        files = list(res_dir.glob("*.parquet"))
        self.assertEqual(len(files), 2)
        for f in files:
            self.assertEqual(len(pd.read_parquet(f)), 3 * BATCH_SIZE)
        self.assertEqual(
            ResultsWriter.count_complete_rows(res_dir), 3 * BATCH_SIZE,
        )

    def test_declining_resume_starts_new_run(self):
        """Answering N starts over in a new dir, keeping the old one."""
        expected = self.reference_run()
        param_file = make_param_file(self.tmp, "resumed")

        self.run_main(param_file, interrupt_at_flush=3)
        interrupted = self.latest_dir("resumed")
        inp = self.run_main(param_file, answer="N")

        inp.assert_called_once()
        self.assertNotEqual(self.latest_dir("resumed"), interrupted)
        self.assert_same_results(self.latest_dir("resumed"), expected)
        self.assertEqual(
            ResultsWriter.count_complete_rows(interrupted), 2 * BATCH_SIZE,
        )

    def test_finished_run_is_not_resumed(self):
        """Re-running a finished scenario starts a new run, no prompt."""
        expected = self.reference_run()
        inp = self.run_main(make_param_file(self.tmp, "reference"))
        inp.assert_not_called()
        self.assertNotEqual(self.latest_dir("reference"), expected)
        self.assert_same_results(self.latest_dir("reference"), expected)

    def test_selection_does_not_depend_on_start_drop(self):
        """Drops select the same satellites wherever the run started."""
        par_file = make_param_file(self.tmp, "start_drop")
        full, tail = MemoryResultsWriter(), MemoryResultsWriter()
        run_simulation(load_params(par_file), full)
        start = 3 * BATCH_SIZE
        run_simulation(load_params(par_file), tail, start_from_drop=start)

        self.assertEqual(full.rows.keys(), tail.rows.keys())
        for attr in full.rows:
            self.assertIn(STR_SEPARATOR, attr)
            self.assertEqual(len(tail.rows[attr]), N_STEPS - start)
            np.testing.assert_array_equal(
                [r[f"i{STR_SEPARATOR}NGSO"] for r in tail.rows[attr]],
                [r[f"i{STR_SEPARATOR}NGSO"] for r in full.rows[attr][start:]],
                err_msg=attr,
            )


if __name__ == "__main__":
    unittest.main()
