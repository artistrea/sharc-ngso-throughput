import itur
import shutil
import argparse
from sharc.satellite.scripts.plot_globe import plot_globe_with_borders
from sharc.p618 import rain_attenuation_inv_ccdf
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, ParametersGSO, STR_SEPARATOR
)
from datetime import datetime
from pathlib import Path
from sharc.parameters.constants import BOLTZMANN_CONSTANT
from sharc.antenna.antenna_1428 import es_ant_gain_1428
from sharc.support.sharc_geom import CoordinateSystem
from sharc.support.geometry import (
    SimulatorGeometry, ENUReferenceFrame, DWNReferenceFrame, plot_geom
)
from sharc.satellite.ngso.orbit_model import OrbitModel
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import csv

RESULTS_DIR = "./results-ngso-slower"
DEBUG = False


class ResultsWriter:
    def __init__(self, directory: str | Path, inp_file: str | Path = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = Path(directory) / timestamp
        self.directory.mkdir(parents=True, exist_ok=False)

        if inp_file:
            inp_file = Path(inp_file)
            shutil.copy2(inp_file, self.directory / inp_file.name)

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


@dataclass
class GSOLinkContext:
    """
    All precomputed, time-invariant quantities for one GSO reference link.
    Built once before the time loop; consumed inside it.
    """
    label: str
    par: ParametersGSO

    # Geometry objects
    global_coord_sys: CoordinateSystem
    global_reference_frame: ENUReferenceFrame
    es_geom: SimulatorGeometry
    ss_geom: SimulatorGeometry

    # Precomputed scalars
    frequency_ghz: float
    lmbda: float
    es_rx_ant_d_lmbda: float
    fspl_dB: float              # scalar — distance doesn't change for GSO
    elevation_deg: float

    # Rain attenuation inverse CCDF callable
    rain_inv_ccdf: object       # callable: p -> attenuation [dB]


def _build_gso_link_context(gso_par: ParametersGSO) -> GSOLinkContext:
    """
    Precompute everything that is static for a given GSO link.
    Mirrors the module-level setup from the original script, but scoped
    to a single ParametersGSO instance.
    """
    es = gso_par.earth_station
    lat, lon = es.lat_deg, es.lon_deg

    # Altitude: derive from topography if not explicitly set
    alt_m = es.alt_m
    if alt_m is None:
        alt_m = itur.topographic_altitude(lat, lon).to(itur.u.m).value

    global_reference_frame = ENUReferenceFrame(lat=lat, lon=lon, alt=alt_m)

    global_coord_sys = CoordinateSystem()
    global_coord_sys.set_reference(lat, lon, alt_m)

    # Earth station geometry (origin of ENU frame)
    es_geom = SimulatorGeometry(1, False, global_reference_frame)
    es_geom.set_global_coords(
        np.array([0.]), np.array([0.]), np.array([0.])
    )

    # GSO satellite geometry
    ss_geom = SimulatorGeometry(1, True, global_reference_frame)
    ss_geom.set_local_reference_frame(
        DWNReferenceFrame(lat=0., lon=gso_par.orbital_slot_deg, alt=35786e3)
    )
    ss_geom.set_local_coords(
        np.array([0.]), np.array([0.]), np.array([0.]),
        np.array([0.]), np.array([0.])
    )

    # Point the earth station antenna toward the satellite
    phi, theta = es_geom.get_global_pointing_vector_to(ss_geom)
    az, el = phi[0], 90. - theta[0]
    es_geom.set_global_coords(azim=az, elev=el)
    ss_geom.set_local_coords(np.array([0.]), np.array([0.]), np.array([0.]))

    elevation_deg = es_geom.get_local_elevation(ss_geom)

    # Link budget precomputations
    freq_ghz = gso_par.center_freq_GHz
    lmbda = 3e8 / (freq_ghz * 1e9)
    es_rx_ant_d_lmbda = gso_par.rx_antenna_size_m / lmbda

    fspl_dB = float(np.ravel(
        20 * np.log10(es_geom.get_3d_distance_to(ss_geom))
        + 20 * np.log10(freq_ghz*1e3)
        - 27.55
    ).item())

    rain_inv_ccdf = rain_attenuation_inv_ccdf(
        lat, lon,
        freq_ghz * itur.u.GHz,
        elevation_deg,
        alt_m * itur.u.m,
    )

    return GSOLinkContext(
        label=gso_par.full_name,
        par=gso_par,
        global_coord_sys=global_coord_sys,
        global_reference_frame=global_reference_frame,
        es_geom=es_geom,
        ss_geom=ss_geom,
        frequency_ghz=freq_ghz,
        lmbda=lmbda,
        es_rx_ant_d_lmbda=es_rx_ant_d_lmbda,
        fspl_dB=fspl_dB,
        elevation_deg=elevation_deg,
        rain_inv_ccdf=rain_inv_ccdf,
    )


def _compute_gso_link_metrics(
    ctx: GSOLinkContext,
    ngso_geom: SimulatorGeometry,
    p_rain: np.ndarray,
    par: ParametersNGSO2GSO,
) -> dict:
    """
    Compute C, C/N, C/(N+I) for one GSO link at one time step.

    Parameters
    ----------
    ctx       : precomputed static context for this GSO link
    ngso_geom : NGSO satellite geometry for the current time instant,
                already expressed in this GSO link's ENU frame
    p_rain    : uniform [0,1] sample(s) for rain attenuation draw
    par       : top-level simulation parameters

    Returns
    -------
    dict with keys: c, cn, cni, gso_rain_att
    """
    gso_par = ctx.par
    ngso_par = par.ngso
    tx = ngso_par.tx_model

    rain_att = ctx.rain_inv_ccdf(p_rain).value[:, 0]

    carrier_pow_dl_per_mhz = (
        gso_par.eirp_dBW_per_carrier - 10*np.log10(gso_par.bandwidth_MHz)
        + gso_par.peak_rx_antenna_gain - gso_par.added_loss
        - ctx.fspl_dB - rain_att
    )

    if tx.model == "CONSTANT_PFD_AT_GND":
        off_axis = ctx.es_geom.get_off_axis_angle(ngso_geom)[0]
        selected = np.argsort(off_axis)[:ngso_par.n_co_channel]

        ant_rx_gain = es_ant_gain_1428(
            off_axis[selected],
            gso_par.peak_rx_antenna_gain,
            ctx.es_rx_ant_d_lmbda,
        )
        # TODO: use NGSO own attenuation
        # challenge will probably be performance. We would first have to
        # precompute everything possible. If it's not enough, interpolate
        # through different elevations
        # TODO: correlation with GSO? Maybe use the same p% of the time for both attenuations
        # or, even more correct, use the same rain rate for both at the same time step
        # NOTE: kinda does not make sense to sample from a random distribution
        # for a single time instant. Think of doing montecarlo for each time step
        interf_pow_per_sat = (
            tx.pfd_at_ref_bandwidth_dBW_m2
            + 10 * np.log10(ctx.lmbda**2 / (4 * np.pi))
            - rain_att
            + ant_rx_gain
        )
        interf_pow = 10 * np.log10(np.sum(10 ** (interf_pow_per_sat / 10)))
    else:
        raise ValueError(f"Unsupported TX model: {tx.model}")

    noise_density = (
        10 * np.log10(BOLTZMANN_CONSTANT)
        + gso_par.peak_rx_antenna_gain
        - gso_par.g_over_t_dB_per_K
    )
    noise_pow = noise_density + 10 * np.log10(par.ref_bandwidth_Hz)

    cn = carrier_pow_dl_per_mhz - noise_pow
    cni = carrier_pow_dl_per_mhz - 10 * np.log10(
        10 ** (noise_pow / 10) + 10 ** (interf_pow / 10)
    )
    epfd = (
        interf_pow - gso_par.peak_rx_antenna_gain
        + 10 * np.log10((4 * np.pi) / ctx.lmbda**2)
    )

    return {
        "c": carrier_pow_dl_per_mhz,
        "cn": cn,
        "cni": cni,
        "i": interf_pow,
        "epfd": epfd,
        "gso_rain_att": rain_att,
    }


def run_simulation(par: ParametersNGSO2GSO, par_file: Path = None):
    # ------------------------------------------------------------------
    # One-time setup: orbits
    # ------------------------------------------------------------------
    orbit_models = [
        OrbitModel(
            Nsp=p.sats_per_plane, Np=p.n_planes,
            phasing=p.phasing_deg, long_asc=p.long_asc_deg,
            omega=p.omega_deg, delta=p.inclination_deg,
            hp=p.perigee_alt_km, ha=p.apogee_alt_km,
            Mo=p.initial_mean_anomaly,
            # IGNORE THIS
            model_time_as_random_variable=False,
            t_min=0, t_max=0,
        )
        for p in par.ngso.orbits
    ]
    total_sats = sum(p.n_planes * p.sats_per_plane for p in par.ngso.orbits)

    # ------------------------------------------------------------------
    # One-time setup: one context object per GSO link
    # ------------------------------------------------------------------
    gso_contexts: list[GSOLinkContext] = [
        _build_gso_link_context(gso_par)
        for gso_par in par.gso_links
    ]

    results_writer = ResultsWriter(f"{RESULTS_DIR}/", par_file)

    timeline = np.arange(par.min_t_s, par.max_t_s, par.delta_t_s)
    seed_seq = np.random.SeedSequence(par.seed)
    child_seed_seq = seed_seq.spawn(len(timeline))

    first_plot = True

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    for timeline_i in tqdm(range(len(timeline))):
        rng = np.random.default_rng(child_seed_seq[timeline_i])
        t = timeline[timeline_i]

        # --- NGSO positions (ECEF, then per-GSO-link ENU) ---
        orbit_positions = [
            o.get_orbit_positions_time_instant(np.array([t]))
            for o in orbit_models
        ]
        ngso_x_ecef = np.ravel(np.concatenate([pos['sx'] for pos in orbit_positions])) * 1e3
        ngso_y_ecef = np.ravel(np.concatenate([pos['sy'] for pos in orbit_positions])) * 1e3
        ngso_z_ecef = np.ravel(np.concatenate([pos['sz'] for pos in orbit_positions])) * 1e3

        # Draw one rain probability sample shared across all GSO links.
        # Using the same p% keeps the comparison fair and is physically
        # motivated (same atmospheric snapshot). See TODO in original.
        p_rain = rng.uniform(0, 1, np.array([t]).shape)

        # --- Per-GSO-link metrics ---
        ngso_geom = SimulatorGeometry(total_sats)
        for ctx in gso_contexts:
            # Project ECEF NGSO positions into this link's ENU frame
            nx, ny, nz = ctx.global_coord_sys.ecef2enu(
                ngso_x_ecef, ngso_y_ecef, ngso_z_ecef
            )
            ngso_geom.set_global_coords(nx, ny, nz)

            metrics = _compute_gso_link_metrics(ctx, ngso_geom, p_rain, par)
            results_writer.add_results(
                metrics, f"gso_per_iteration{STR_SEPARATOR}{ctx.label}"
            )

        # --- Debug plot (first step only) ---
        if DEBUG and first_plot:
            first_plot = False
            # Use the first GSO context for the reference plot
            ctx0 = gso_contexts[0]
            fig = plot_globe_with_borders(True, ctx0.global_coord_sys, False)
            plot_geom(fig, ctx0.es_geom, plot_pointing=True, boresight_length=100 * 1e5)
            plot_geom(fig, ctx0.ss_geom)
            plot_geom(fig, ngso_geom)
            fig.show()

        if timeline_i % par.batch_size == 0:
            results_writer.flush_results()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    results_writer.flush_results()
    results_writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="SHARC - Radio Sharing and Compatiblity Monte Carlo Simulator"
    )
    parser.add_argument("-p", "--param-file", help="Path to parameter file")
    args = parser.parse_args()

    if Path(args.param_file).is_absolute():
        param_file = Path(args.param_file)
    else:
        param_file = Path(".") / args.param_file

    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(param_file)
    par.validate("ngso2gso")

    run_simulation(
        par, param_file
    )


if __name__ == "__main__":
    main()
