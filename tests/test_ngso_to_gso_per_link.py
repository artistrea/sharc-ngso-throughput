"""
Regression tests for ngso_to_gso: interference into a GSO link must be
computed from that link's own earth station antenna pointing.

GSO links at the same earth station are grouped so NGSO positions are only
converted once per station. A bug computed off-axis angles (rx antenna gain
and GSO arc avoidance) once per group, using the pointing of the group's
first link, so every other link in the group got wrong interference.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from sharc.ngso_to_gso import run_simulation
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR,
)

EARTH_STATION = {"label": "Las Vegas", "lat_deg": 36.19, "lon_deg": -115.18}

# Two GSO satellites far apart in the arc, seen from the same earth station
LINK_A = {
    "label": "A", "orbital_slot_deg": -102.75,
    "center_freq_GHz": 19.95, "bandwidth_MHz": 36.0,
    "eirp_dBW_per_carrier": 59.5, "peak_rx_antenna_gain": 65.4,
    "rx_antenna_size_m": 13.2, "g_over_t_dB_per_K": 41.4,
    "station_type": "GW",
}
LINK_B = {**LINK_A, "label": "B", "orbital_slot_deg": -67.0}


class MemoryResultsWriter:
    """Stand-in for ResultsWriter that keeps results in memory."""

    def __init__(self):
        self.rows = {}

    def add_results(self, new_res: dict, attr_name: str):
        self.rows.setdefault(attr_name, []).append(dict(new_res))

    def flush_results(self):
        pass

    def close(self, *args, **kwargs):
        pass

    def column(self, link_label: str, col: str) -> np.ndarray:
        attr = (
            f"gso_per_iteration{STR_SEPARATOR}{link_label}"
            f"{STR_SEPARATOR}{EARTH_STATION['label']}"
        )
        return np.array([r[col] for r in self.rows[attr]], dtype=float)


def make_params(tmp_dir: Path, links: list[dict]) -> ParametersNGSO2GSO:
    raw = {"ngso2gso": {
        "version": 2,
        "scenario_name": "per_link_test",
        "seed": 7,
        "delta_t_s": 10.0,
        "min_t_s": 0.0,
        "max_t_s": 200.0,
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
        "gso_links": [
            {**link, "earth_station": dict(EARTH_STATION)} for link in links
        ],
    }}
    path = tmp_dir / f"params_{len(list(tmp_dir.iterdir()))}.yaml"
    path.write_text(yaml.safe_dump(raw))
    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(path)
    par.validate("ngso2gso")
    return par


def run(tmp_dir: Path, links: list[dict]) -> MemoryResultsWriter:
    writer = MemoryResultsWriter()
    run_simulation(make_params(tmp_dir, links), writer)
    return writer


class NGSO2GSOPerLinkOffAxisTest(unittest.TestCase):
    """Interference into a link must not depend on the other links."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cls.alone_a = run(d, [LINK_A])
            cls.alone_b = run(d, [LINK_B])
            cls.a_then_b = run(d, [LINK_A, LINK_B])
            cls.b_then_a = run(d, [LINK_B, LINK_A])

    def _assert_same(self, got, expected, link):
        for col in ("i", "epfd", "cni", f"i{STR_SEPARATOR}NGSO"):
            np.testing.assert_allclose(
                got.column(link, col), expected.column(link, col),
                err_msg=f"link {link}, column {col}",
            )

    def test_scenario_has_interference(self):
        """Sanity check: the scenario actually produces interference."""
        for res, link in ((self.alone_a, "A"), (self.alone_b, "B")):
            i = res.column(link, "i")
            self.assertTrue(np.all(np.isfinite(i)), f"link {link}: {i}")

    def test_links_see_different_interference(self):
        """Sanity check: the two links are distinguishable at all."""
        self.assertFalse(np.allclose(
            self.alone_a.column("A", "i"), self.alone_b.column("B", "i"),
        ))

    def test_second_link_at_station_matches_running_alone(self):
        self._assert_same(self.a_then_b, self.alone_b, "B")
        self._assert_same(self.b_then_a, self.alone_a, "A")

    def test_first_link_at_station_matches_running_alone(self):
        self._assert_same(self.a_then_b, self.alone_a, "A")
        self._assert_same(self.b_then_a, self.alone_b, "B")


if __name__ == "__main__":
    unittest.main()
