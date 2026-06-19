import itur

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from sharc.parameters.constants import BOLTZMANN_CONSTANT
import csv
import plotly.graph_objects as go
from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.parameters.parameters_orbit import ParametersOrbit
from sharc.satellite.scripts.plot_globe import plot_globe_with_borders
from sharc.p618 import rain_attenuation_inv_ccdf
from sharc.support.sharc_geom import CoordinateSystem
from sharc.support.geometry import (
    SimulatorGeometry, ENUReferenceFrame, DWNReferenceFrame, plot_geom
)

import numpy as np

DEBUG = True

SEED = 1251
Nco = 2
DELTA_T = 1
MIN_T = 0
MAX_T = int(1e4)
# TODO: optimize for batch processing
BATCH_SIZE = 100
REF_BANDWIDTH = 1e6

AVAILABLE_INTERF_TX_MODELS = [
    "CONSTANT_PFD_AT_GND",
    # some other, more realistic?
]
TX_MODEL = "CONSTANT_PFD_AT_GND"
TX_MODELS_INFO = {
    "CONSTANT_PFD_AT_GND": {
        "PFD_at_ref_bandwidth": -113
    }
}

AVAILABLE_REFERENCE_GSO_LINKS = [
    "DirecTV 15 GW", "Jupiter 97W GW", "Jupiter 97W CT",
    "Galaxy 30 GW", "Galaxy 30 CT", "SES-15 GW",
    "SES-17 GW", "SES-17 CT", "Viasat-IOM GW",
    "Viasat-IOM CT",
]

REFERENCE_GSO_LINK_INFO = {
    "DirecTV 15 GW": {  # checked
        "orbital_slot_deg": -102.75, "center_freq_GHz": 19.95,
        "downlink_freq_MHz": (18300, 20200),
        "bandwidth_MHz": 36, "eirp_dBW_per_carrier": 59.5,
        "rx_antenna_size_m": 13.2, "g_over_t_dB_per_K": 41.4,
        "peak_rx_antenna_gain": 65.4,
        "station_type": "GW",
    },
    "Jupiter 97W GW": {  # TODO: check
        "orbital_slot_deg": -97.1, "downlink_freq_MHz": (19700, 20200),
        "bandwidth_MHz": 250, "eirp_dBW_per_carrier": 61.0,
        "antenna_size_m": 6.3, "g_over_t_dB_per_K": 38.0,
        "station_type": "GW",
    },
    "Jupiter 97W CT": {  # TODO: check
        "orbital_slot_deg": -97.1, "center_freq_GHz": 18.675,
        "downlink_freq_GHz": (18.3, 20.2), "bandwidth_MHz": 500,
        "eirp_dBW_per_carrier": 64.0, "antenna_size_m": 0.99,
        "g_over_t_dB_per_K": 19.7, "station_type": "CT",
    },
    "Galaxy 30 GW": {  # TODO: check
        "orbital_slot_deg": -125.0, "downlink_freq_MHz": (17800, 20200),
        "bandwidth_MHz": 1405, "eirp_dBW_per_carrier": 43.0,
        "antenna_size_m": 9.4, "g_over_t_dB_per_K": 38.0,
        "station_type": "GW",
    },
    "Galaxy 30 CT": {  # TODO: check
        "orbital_slot_deg": -125.0, "center_freq_GHz": 19.95,
        "downlink_freq_GHz": (18.3, 20.2), "bandwidth_MHz": 724,
        "eirp_dBW_per_carrier": -61.4, "antenna_size_m": 0.98,
        "g_over_t_dB_per_K": 20.1, "station_type": "CT",
    },
    "SES-15 GW": {  # TODO: check
        "orbital_slot_deg": -129.15, "downlink_freq_MHz": (18300, 20200),
        "bandwidth_MHz": 225, "eirp_dBW_per_carrier": 63.3,
        "antenna_size_m": 9.4, "g_over_t_dB_per_K": 38.0,
        "station_type": "GW",
    },
    "SES-17 GW": {  # TODO: check
        "orbital_slot_deg": -67.0, "downlink_freq_MHz": (17800, 20200),
        "bandwidth_MHz": 250, "eirp_dBW_per_carrier": 45.2,
        "antenna_size_m": 9.4, "g_over_t_dB_per_K": 41.3,
        "station_type": "GW",
    },
    "SES-17 CT": {  # TODO: check
        "orbital_slot_deg": -67.0, "center_freq_GHz": 19.95,
        "downlink_freq_GHz": (18.3, 20.2), "bandwidth_MHz": 416,
        "eirp_dBW_per_carrier": 67.0, "antenna_size_m": 0.97,
        "g_over_t_dB_per_K": 18.7, "station_type": "CT",
    },
    "Viasat-IOM GW": {  # TODO: check
        "orbital_slot_deg": -115.1, "downlink_freq_MHz": (18300, 20200),
        "bandwidth_MHz": 500, "eirp_dBW_per_carrier": 64.0,
        "antenna_size_m": 7.3, "g_over_t_dB_per_K": 38.8,
        "station_type": "GW",
    },
    "Viasat-IOM CT": {  # TODO: check
        "orbital_slot_deg": -115.1, "center_freq_GHz": 19.95,
        "downlink_freq_GHz": (18.3, 20.2), "bandwidth_MHz": 416,
        "eirp_dBW_per_carrier": -61.0, "antenna_size_m": 0.745,
        "g_over_t_dB_per_K": 16.4, "station_type": "CT",
    },
}

REFERENCE_GSO_LINK = "DirecTV 15 GW"
RESULTS_DIR = "./results-DirecTV-15-GW"


class ResultsWriter:
    def __init__(self, directory):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = Path(directory) / timestamp
        self.directory.mkdir(parents=True, exist_ok=False)

        self.res = {}
        self.files = {}
        self.writers = {}

    def add_results(self, new_res: dict, attr_name: str):
        # Normalize everything to lists
        normalized = {}
        for k, v in new_res.items():
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    normalized[k] = [v.item()]
                else:
                    normalized[k] = v.tolist()
            elif np.isscalar(v):
                normalized[k] = [v]
            else:
                normalized[k] = list(v)

        if attr_name not in self.res:
            self.res[attr_name] = normalized
            return

        provided_keys = list(normalized.keys())
        saved_keys = list(self.res[attr_name].keys())

        if (
            any(s not in provided_keys for s in saved_keys)
            or any(p not in saved_keys for p in provided_keys)
        ):
            raise ValueError("You must maintain a specific runtime schema.")

        for k in provided_keys:
            self.res[attr_name][k].extend(normalized[k])

    def flush_results(self):
        for attr_name, data in self.res.items():
            lengths = {len(v) for v in data.values()}
            if len(lengths) != 1:
                raise ValueError(f"Inconsistent column lengths for '{attr_name}'.")

            if not data:
                continue

            n_rows = len(next(iter(data.values())))
            if n_rows == 0:
                continue

            # Open file lazily
            if attr_name not in self.files:
                path = self.directory / f"{attr_name}.csv"
                is_new = not path.exists()

                f = open(path, "a", newline="")
                writer = csv.writer(f)

                if is_new:
                    writer.writerow(data.keys())

                self.files[attr_name] = f
                self.writers[attr_name] = writer

            writer = self.writers[attr_name]

            for row in zip(*data.values()):
                writer.writerow(row)

            self.files[attr_name].flush()

            # Clear buffered results while preserving schema
            for k in data:
                data[k].clear()

    def close(self):
        for f in self.files.values():
            f.close()

        self.files.clear()
        self.writers.clear()


gso_link_info = REFERENCE_GSO_LINK_INFO[REFERENCE_GSO_LINK]
# Las Vegas, NV (36.19° N, 115.18° W),
# Kansas City, MO (39.07° N, 94.60° W),
# Miami, FL (25.77° N, 80.19° W)
gso_es_pos = (36.19, -115.18)
gso_es_alt = itur.topographic_altitude(
    gso_es_pos[0], gso_es_pos[1]
).to(itur.u.m).value
print("gso_es_alt", gso_es_alt)

# gso earth station at (0,0,0)
global_reference_frame = ENUReferenceFrame(
    lat=gso_es_pos[0], lon=gso_es_pos[1], alt=gso_es_alt
)


def plot_gso():
    fig = plot_globe_with_borders(True, global_coord_sys, False)
    # plot_geom(fig, gso_es_geom, plot_pointing=True)
    plot_geom(fig, gso_es_geom, plot_pointing=True, boresight_length=100*1e5)
    plot_geom(fig, gso_ss_geom)
    # Set the camera position in Plotly
    # show_range = 1e4
    show_range = 8e7
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="cube",
            zaxis=dict(
                range=(-show_range / 2, show_range / 2)
            ),
            yaxis=dict(
                range=(-show_range / 2, show_range / 2)
            ),
            xaxis=dict(
                range=(-show_range / 2, show_range / 2)
            ),
            camera=dict(
                # center=dict(x=0, y=0, z=center_of_earth.geom.z_global[0] / show_range / 1e3),
                # eye=dict(x=0, y=0, z=0.7),  # Eye position (above the center)
                # up=dict(x=0, y=1, z=0)      # "Up" is along +y (default is usually +z)
            )
        ),
        legend=dict(
            x=0.02,        # Move to left
            y=0.02,        # Near top
            bgcolor='rgba(255,255,255,1)',  # Optional: semi-transparent background
            bordercolor='black',
            borderwidth=1
        ),
        # width=700,
        # height=700,
    )
    return fig

# NOTE:
# coordinate system is kinda legacy code already. Still, useful for plotting
# and simpler
global_coord_sys = CoordinateSystem()
global_coord_sys.set_reference(
    gso_es_pos[0], gso_es_pos[1], gso_es_alt
)

gso_es_geom = SimulatorGeometry(
    1, False, global_reference_frame
)

gso_ss_geom = SimulatorGeometry(
    1, True, global_reference_frame
)
gso_ss_geom.set_local_reference_frame(
    DWNReferenceFrame(
        lat=0., lon=gso_link_info['orbital_slot_deg'], alt=35786e3
    )
)
gso_ss_geom.set_local_coords(
    np.array([0.]), np.array([0.]), np.array([0.]),
    np.array([0.]), np.array([0.])
)
gso_es_geom.set_global_coords(
    np.array([0.]), np.array([0.]), np.array([0.]),
)

phi, theta = gso_es_geom.get_global_pointing_vector_to(gso_ss_geom)
az, el = phi[0], 90. - theta[0]
gso_es_geom.set_global_coords(azim=az, elev=el)
gso_ss_geom.set_local_coords(np.array([0.]), np.array([0.]), np.array([0.]))

gso_link_elevation = gso_es_geom.get_local_elevation(gso_ss_geom)

gso_link_rain_inv_ccdf = rain_attenuation_inv_ccdf(
    gso_es_pos[0], gso_es_pos[1],
    gso_link_info['center_freq_GHz'] * itur.u.GHz,
    gso_link_elevation,
    gso_es_alt * itur.u.m,
)


def es_ant_gain_1428(off_axis, ant_gain, d_over_lmbda):
    off_axis = np.atleast_1d(off_axis)
    g = np.zeros_like(off_axis)

    if d_over_lmbda < 20:
        raise ValueError("ma burro")
    elif d_over_lmbda <= 25:
        g_max = 20*np.log10(d_over_lmbda) + 7.7
        g1 = 29 - 25 * np.log10(95/d_over_lmbda)
        phi_m = 20 / d_over_lmbda * np.sqrt(g_max - g1)
        g[off_axis < phi_m] = ant_gain - 2.5e-3 * (d_over_lmbda*off_axis[off_axis < phi_m])**2
        g[off_axis < 95 / d_over_lmbda] = g1
        g[off_axis < 33.1] = 29 - 25 * np.log10(off_axis[off_axis < 33.1])
        g[off_axis < 80.] = -9
        g[off_axis < 180.] = -5
        # unsupported:
        g[off_axis > 180.] = -5
    elif d_over_lmbda <= 100:
        g_max = 20*np.log10(d_over_lmbda) + 7.7
        g1 = 29 - 25 * np.log10(95/d_over_lmbda)
        phi_m = 20 / d_over_lmbda * np.sqrt(g_max - g1)
        g[off_axis < phi_m] = ant_gain - 2.5e-3 * (d_over_lmbda*off_axis[off_axis < phi_m])**2
        g[off_axis < 95 / d_over_lmbda] = g1
        g[off_axis < 33.1] = 29 - 25 * np.log10(off_axis[off_axis < 33.1])
        g[off_axis < 80.] = -9
        g[off_axis < 120.] = -4
        g[off_axis < 180.] = -9
        # unsupported:
        g[off_axis > 180.] = -9
    elif d_over_lmbda > 100:
        g_max = 20*np.log10(d_over_lmbda) + 8.4
        g1 = -1 + 15 * np.log10(d_over_lmbda)
        phi_m = 20 / d_over_lmbda * np.sqrt(g_max - g1)
        phi_r = 15.85 * d_over_lmbda ** -0.6
        g[off_axis < phi_m] = ant_gain - 2.5e-3 * (d_over_lmbda*off_axis[off_axis < phi_m])**2
        g[off_axis < phi_r] = g1
        g[off_axis < 10.] = 29 - 25 * np.log10(off_axis[off_axis < 10.])
        g[off_axis < 34.1] = 34 - 30 * np.log10(off_axis[off_axis < 34.1])
        g[off_axis < 80.] = -12
        g[off_axis < 120.] = -7
        g[off_axis < 180.] = -12
        # unsupported:
        g[off_axis > 180.] = -12

    return g


if __name__ == "__main__":
    orbit_params = [
        ParametersOrbit(
            n_planes=289, sats_per_plane=4, inclination_deg=51.9, perigee_alt_km=630., apogee_alt_km=630.,
            # DOES NOT SPECIFY:
            phasing_deg=7.5, long_asc_deg=0.,
            omega_deg=0.,  # argument of perigee. Doesn't matter for circular orbit
            initial_mean_anomaly=0.,
        ),
        ParametersOrbit(
            n_planes=1292, sats_per_plane=1, inclination_deg=42., perigee_alt_km=610, apogee_alt_km=610,
            # DOES NOT SPECIFY:
            phasing_deg=7.5, long_asc_deg=0., omega_deg=0., initial_mean_anomaly=0.,
        ),
        ParametersOrbit(
            n_planes=782, sats_per_plane=1, inclination_deg=33., perigee_alt_km=590, apogee_alt_km=590,
            # DOES NOT SPECIFY:
            phasing_deg=7.5, long_asc_deg=0., omega_deg=0., initial_mean_anomaly=0.,
        ),
        ParametersOrbit(
            n_planes=1, sats_per_plane=2, inclination_deg=30., perigee_alt_km=590, apogee_alt_km=590,
            # DOES NOT SPECIFY:
            phasing_deg=7.5, long_asc_deg=0., omega_deg=0., initial_mean_anomaly=0.,
        ),
    ]
    orbits = [
        OrbitModel(
            Nsp=p.sats_per_plane, Np=p.n_planes,
            phasing=p.phasing_deg, long_asc=p.long_asc_deg,
            omega=p.omega_deg, delta=p.inclination_deg,
            hp=p.perigee_alt_km, ha=p.apogee_alt_km,
            Mo=p.initial_mean_anomaly,
            # we don't use time as random, we sample at each instant we want
            model_time_as_random_variable=False,
            t_min=0,
            t_max=0,
        ) for p in orbit_params
    ]
    total_sats = sum(o.n_planes * o.sats_per_plane for o in orbit_params)

    assert total_sats == 3232

    frequency_ghz = gso_link_info['center_freq_GHz']
    lmbda = 3e8 / (frequency_ghz*1e9)
    es_rx_ant_d_lmbda = gso_link_info['rx_antenna_size_m'] / lmbda
    gso_fspl = np.ravel(
        20 * np.log10(
            gso_es_geom.get_3d_distance_to(gso_ss_geom)
        ) + 20 * np.log10(frequency_ghz) - 27.55
    )

    first_plot = True
    rng = np.random.default_rng(SEED)
    ngso_geom = SimulatorGeometry(total_sats)

    results_writer = ResultsWriter(RESULTS_DIR)

    timeline = np.arange(MIN_T, MAX_T, DELTA_T)
    for timeline_i in tqdm(range(len(timeline))):
        t = timeline[timeline_i]
        orbit_positions = [o.get_orbit_positions_time_instant(np.array([t])) for o in orbits]
        # ECEF
        ngso_sat_x = np.ravel(np.concatenate([pos['sx'] for pos in orbit_positions])) * 1e3
        ngso_sat_y = np.ravel(np.concatenate([pos['sy'] for pos in orbit_positions])) * 1e3
        ngso_sat_z = np.ravel(np.concatenate([pos['sz'] for pos in orbit_positions])) * 1e3
        # ECEF2ENU (considering victim earth station ENU)
        ngso_sat_x, ngso_sat_y, ngso_sat_z = \
            global_coord_sys.ecef2enu(ngso_sat_x, ngso_sat_y, ngso_sat_z)

        ngso_geom.set_global_coords(ngso_sat_x, ngso_sat_y, ngso_sat_z)

        p = rng.uniform(0, 1, t.shape)
        gso_rain_att = gso_link_rain_inv_ccdf(p).value[:, 0]

        carrier_pow_dl = (
            gso_link_info['eirp_dBW_per_carrier']
            + gso_link_info['peak_rx_antenna_gain'] - 3.0
            - gso_fspl - gso_rain_att
        )
        if TX_MODEL == "CONSTANT_PFD_AT_GND":
            interf_pfd = TX_MODELS_INFO[TX_MODEL]["PFD_at_ref_bandwidth"]
            # TODO: use NGSO own attenuation
            # challenge will probably be performance. We would first have to
            # precompute everything possible. If it's not enough, interpolate
            # through different elevations
            # TODO: correlation with GSO? Maybe use the same p% of the time for both attenuations
            # or, even more correct, use the same rain rate for both at the same time step
            # NOTE: kinda does not make sense to sample from a random distribution
            # for a single time instant. Think of doing montecarlo for each time step
            ngso_rain_att = gso_rain_att
            off_axis = gso_es_geom.get_off_axis_angle(ngso_geom)[0]
            # WORST CASE
            selected_sats = np.argsort(off_axis)[:Nco]

            ant_rx_gain = es_ant_gain_1428(
                off_axis[selected_sats], gso_link_info["peak_rx_antenna_gain"], es_rx_ant_d_lmbda
            )
            interf_pow_per_sat = (
                interf_pfd + 10*np.log10(lmbda**2/(4*np.pi)) - ngso_rain_att + ant_rx_gain
            )
            interf_pow = 10 * np.log10(np.sum(10**(interf_pow_per_sat/10)))
        else:
            raise ValueError(f"Cannot deal with TX_MODEL={TX_MODEL}")

        noise_density = (
            10 * np.log10(BOLTZMANN_CONSTANT)
            + gso_link_info["peak_rx_antenna_gain"]
            - gso_link_info["g_over_t_dB_per_K"]
        )
        noise_pow = noise_density + 10 * np.log10(REF_BANDWIDTH)
        cn = carrier_pow_dl - noise_pow
        cni = carrier_pow_dl - 10*np.log10(
            10**(noise_pow/10) + 10**(interf_pow/10)
        )
        results_writer.add_results({
            "c": carrier_pow_dl,
            "cn": cn,
            "cni": cni,
            "gso_rain_att": gso_rain_att,
        }, "gso_per_iteration")

        if DEBUG and first_plot:
            first_plot = False
            fig = plot_gso()
            # fig = plot_globe_with_borders(True, global_coord_sys, False)
            plot_geom(fig, ngso_geom)
            fig.show()
        if timeline_i % BATCH_SIZE == 0:
            results_writer.flush_results()
