"""
Tests for ngso2gso.band_overlap.

Scaling NGSO interference by frequency overlap must only happen when the
user asks for it (band_overlap: FRACTION), and the factors applied must be
recorded with the results.
"""
import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import yaml

from sharc.ngso_to_gso import run_simulation, write_band_overlap
from sharc.parameters.parameters_ngso_to_gso import ParametersNGSO2GSO
from tests.test_ngso_to_gso_per_link import (
    EARTH_STATION, LINK_A, MemoryResultsWriter,
)

# LINK_A receives 19932-19968 MHz (19.95 GHz, 36 MHz).
# This band covers its lower half: 19932-19950 MHz.
HALF_BAND = {"center_freq_GHz": 19.941, "bandwidth_MHz": 18.0}
# Entirely below LINK_A's band.
NO_OVERLAP_BAND = {"center_freq_GHz": 19.0, "bandwidth_MHz": 100.0}

BASE = {"ngso2gso": {
    "version": 2,
    "scenario_name": "band_overlap_test",
    "seed": 7,
    "delta_t_s": 10.0,
    "min_t_s": 0.0,
    "max_t_s": 50.0,
    "batch_size": 10,
    "minimum_elevation": 20.0,
    "selection_strategy": "WC",
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
    "gso_links": [{**LINK_A, "earth_station": dict(EARTH_STATION)}],
}}
LINK_FULL_NAME = "A---" + EARTH_STATION["label"]


class BandOverlapCase:
    """Builds and validates parameter files for one test."""

    def __init__(self, tmp_dir: Path):
        """Write parameter files under tmp_dir."""
        self.tmp_dir = tmp_dir
        self.n = 0

    def params(self, top: dict | None = None, ngso: dict | None = None):
        """Load BASE with top-level and NGSO-system keys overridden."""
        raw = copy.deepcopy(BASE)
        raw["ngso2gso"].update(top or {})
        raw["ngso2gso"]["ngso_systems"][0].update(ngso or {})
        self.n += 1
        path = self.tmp_dir / f"params_{self.n}.yaml"
        path.write_text(yaml.safe_dump(raw))
        par = ParametersNGSO2GSO()
        par.load_parameters_from_file(path)
        par.validate("ngso2gso")
        return par


class NGSO2GSOBandOverlapValidationTest(unittest.TestCase):
    """Which combinations of band_overlap inputs are accepted."""

    def setUp(self):
        """Use a fresh temporary directory per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.case = BandOverlapCase(Path(self._tmp.name))

    def tearDown(self):
        """Remove the temporary directory."""
        self._tmp.cleanup()

    def test_default_is_full_without_warnings(self):
        """No band and no mode: FULL, factor 1, nothing to warn about."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            par = self.case.params()
        self.assertEqual(par.band_overlap, "FULL")
        self.assertEqual(
            par.band_overlap_fractions(), {LINK_FULL_NAME: {"NGSO": 1.0}}
        )

    def test_full_rejects_ngso_band(self):
        """A band set while in FULL mode would be silently ignored."""
        with self.assertRaisesRegex(ValueError, "band_overlap is 'FULL'"):
            self.case.params(ngso=HALF_BAND)

    def test_full_rejects_allow_zero(self):
        """allow_zero_band_overlap means nothing in FULL mode."""
        with self.assertRaisesRegex(ValueError, "allow_zero_band_overlap"):
            self.case.params(top={"allow_zero_band_overlap": True})

    def test_fraction_requires_ngso_band(self):
        """FRACTION needs every system's band."""
        with self.assertRaisesRegex(ValueError, "must set center_freq_GHz"):
            self.case.params(top={"band_overlap": "FRACTION"})

    def test_fraction_computes_fraction(self):
        """FRACTION gives the covered fraction of the victim band."""
        par = self.case.params(
            top={"band_overlap": "FRACTION"}, ngso=HALF_BAND
        )
        self.assertAlmostEqual(
            par.band_overlap_fractions()[LINK_FULL_NAME]["NGSO"], 0.5
        )

    def test_fraction_rejects_zero_overlap(self):
        """Zero overlap is rejected by default."""
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            self.case.params(
                top={"band_overlap": "FRACTION"}, ngso=NO_OVERLAP_BAND
            )

    def test_fraction_allows_zero_overlap_when_asked(self):
        """allow_zero_band_overlap accepts zero overlap, with a warning."""
        with self.assertWarnsRegex(UserWarning, "no frequency overlap"):
            par = self.case.params(
                top={"band_overlap": "FRACTION",
                     "allow_zero_band_overlap": True},
                ngso=NO_OVERLAP_BAND,
            )
        self.assertEqual(
            par.band_overlap_fractions()[LINK_FULL_NAME]["NGSO"], 0.0
        )

    def test_unknown_mode_rejected(self):
        """Only FULL and FRACTION are valid."""
        with self.assertRaisesRegex(ValueError, "must be one of"):
            self.case.params(top={"band_overlap": "PARTIAL"})

    def test_write_band_overlap(self):
        """The factors applied are saved next to the results."""
        par = self.case.params(
            top={"band_overlap": "FRACTION",
                 "allow_zero_band_overlap": True},
            ngso=NO_OVERLAP_BAND,
        )
        path = write_band_overlap(self.case.tmp_dir, par)
        data = json.loads(path.read_text())
        self.assertEqual(data["band_overlap"], "FRACTION")
        self.assertEqual(
            data["links"][LINK_FULL_NAME]["NGSO"],
            {"fraction": 0.0, "scale_dB": None},
        )


class NGSO2GSOBandOverlapSimulationTest(unittest.TestCase):
    """The mode changes the simulated interference as documented."""

    @classmethod
    def setUpClass(cls):
        """Run the same scenario in FULL and in FRACTION (half overlap)."""
        with tempfile.TemporaryDirectory() as d:
            case = BandOverlapCase(Path(d))
            cls.full = MemoryResultsWriter()
            run_simulation(case.params(), cls.full)
            cls.half = MemoryResultsWriter()
            run_simulation(
                case.params(top={"band_overlap": "FRACTION"}, ngso=HALF_BAND),
                cls.half,
            )

    def test_half_overlap_is_3db_lower(self):
        """Half the victim band covered gives interference 3 dB lower."""
        i_full = self.full.column("A", "i")
        i_half = self.half.column("A", "i")
        self.assertTrue(np.all(np.isfinite(i_full)))
        np.testing.assert_allclose(
            i_half, i_full + 10 * np.log10(0.5), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
