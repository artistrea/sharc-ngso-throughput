import itur
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
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

RESULTS_DIR = "./scenarios/results"
RESULTS_JSON_RESUME_PATH = Path(f"{RESULTS_DIR}/results.json")

DEBUG = False


def plot_scenario(
    global_coord_sys: CoordinateSystem,
    gso_es_geom: SimulatorGeometry,
    gso_ss_geom: SimulatorGeometry,
    ngso_geom: SimulatorGeometry,
):
    fig = plot_globe_with_borders(True, global_coord_sys, False)
    # plot_geom(fig, gso_es_geom, plot_pointing=True)
    plot_geom(fig, gso_es_geom, plot_pointing=True, boresight_length=100*1e5)
    plot_geom(fig, gso_ss_geom)
    plot_geom(fig, ngso_geom)
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


class MockResultsWriter:
    def __init__(self, *args):
        pass

    @property
    def directory(self):
        return "mock_dir"

    def add_results(self, *args):
        pass

    def flush_results(self):
        pass

    def close(self):
        pass


class ResultsWriter:
    @staticmethod
    def form_results_path(
        directory: str | Path,
        timestamp: str,
        scenario_name: str,
    ):
        return Path(directory) / f"{timestamp}/{scenario_name}"

    def __init__(
        self,
        directory: str | Path,
        inp_file: str | Path,
        scenario_name: str,
        continue_from_timestamp: str | None = None,
    ):
        if continue_from_timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = continue_from_timestamp

        self.directory = ResultsWriter.form_results_path(
            directory,
            timestamp,
            scenario_name,
        )

        if continue_from_timestamp is None:
            self._setup_new_dir(inp_file)
        else:
            self._check_existing_dir()

        # Buffered results:
        #
        # {
        #     "gso_per_iteration|link_a": {
        #         "c": [...],
        #         "cn": [...],
        #         "cni": [...],
        #     },
        #     "gso_per_iteration|link_b": {
        #         "c": [...],
        #         "cn": [...],
        #         "cni": [...],
        #     },
        # }
        self.res = {}

        # One ParquetWriter per attr_name.
        self.writers = {}

        # One Arrow schema per attr_name.
        self.schemas = {}

    def _check_existing_dir(self):
        if not self.directory.exists():
            raise ValueError(
                "Cannot send results to a non existing results directory "
                f"{self.directory}"
            )

    def _setup_new_dir(self, inp_file: str | Path):
        self.directory.mkdir(parents=True, exist_ok=False)

        inp_file = Path(inp_file)
        shutil.copy2(inp_file, self.directory / inp_file.name)

    def add_results(self, new_res: dict, attr_name: str):
        # Normalize everything to lists.
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
            raise ValueError(
                "You must maintain a specific runtime schema."
            )

        for k in provided_keys:
            self.res[attr_name][k].extend(normalized[k])

    def flush_results(self):
        for attr_name, data in self.res.items():
            lengths = {len(v) for v in data.values()}

            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent column lengths for '{attr_name}'."
                )

            if not data:
                continue

            n_rows = len(next(iter(data.values())))

            if n_rows == 0:
                continue

            # Open Parquet file lazily.
            if attr_name not in self.writers:
                path = self.directory / f"{attr_name}.parquet"

                df = pd.DataFrame(data)

                table = pa.Table.from_pandas(
                    df,
                    preserve_index=False,
                )

                self.schemas[attr_name] = table.schema

                self.writers[attr_name] = pq.ParquetWriter(
                    path,
                    schema=self.schemas[attr_name],
                    compression="zstd",
                )

            # Convert this buffered batch using the established schema.
            df = pd.DataFrame(data)

            table = pa.Table.from_pandas(
                df,
                schema=self.schemas[attr_name],
                preserve_index=False,
            )

            self.writers[attr_name].write_table(table)

            # Clear buffered results while preserving schema.
            for k in data:
                data[k].clear()

    def close(self):
        self.flush_results()

        for writer in self.writers.values():
            writer.close()

        self.writers.clear()
        self.schemas.clear()


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


def _build_gso_link_context(
    gso_par: ParametersGSO,
    coord: CoordinateSystem | None = None
) -> GSOLinkContext:
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

    if coord is not None:
        global_coord_sys = coord
        global_reference_frame = ENUReferenceFrame(
            lat=coord.ref_lat, lon=coord.ref_long, alt=coord.ref_alt
        )
    else:
        global_coord_sys = CoordinateSystem()
        global_coord_sys.set_reference(
            lat, lon, alt_m
        )
        global_reference_frame = ENUReferenceFrame(
            lat=lat, lon=lon, alt=alt_m
        )

    local_reference_frame = ENUReferenceFrame(
        lat=lat, lon=lon, alt=alt_m
    )

    # Earth station geometry (origin of ENU frame)
    es_geom = SimulatorGeometry(1, True, global_reference_frame)
    es_geom.set_local_reference_frame(local_reference_frame)
    es_geom.set_local_coords(
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


@dataclass
class EarthStationSnapshot:
    """Per-earth-station computed quantities for one time step."""
    ngso_geom: SimulatorGeometry
    off_axis: np.ndarray        # shape: (total_sats,)
    selected: np.ndarray        # indices of Nco closest sats
    ant_rx_gain: np.ndarray     # shape: (Nco,) — NOTE: depends on per-link antenna params


def _group_contexts_by_earth_station(
    gso_contexts: list[GSOLinkContext],
) -> dict[str, list[GSOLinkContext]]:
    """
    Group GSO link contexts by earth station identity.
    Earth stations are considered identical if they share the same
    (lat, lon) rounded to 4 decimal places (~11m precision).
    """
    def es_key(ctx: GSOLinkContext) -> str:
        es = ctx.par.earth_station
        return f"{es.lat_deg:.4f},{es.lon_deg:.4f}"

    # Preserve insertion order within groups
    groups: dict[str, list[GSOLinkContext]] = {}
    for ctx in gso_contexts:
        key = es_key(ctx)
        groups.setdefault(key, []).append(ctx)
    return groups


def _compute_gso_link_metrics(
    ctx: GSOLinkContext,
    off_axis: np.ndarray,
    selected: np.ndarray,
    ant_rx_gain: np.ndarray,
    rain_att: float,            # scalar for this time step, precomputed by caller
    par: ParametersNGSO2GSO,
) -> dict:
    gso_par = ctx.par
    tx = par.ngso.tx_model

    carrier_pow_dl_per_mhz = (
        gso_par.eirp_dBW_per_carrier - 10 * np.log10(gso_par.bandwidth_MHz)
        + gso_par.peak_rx_antenna_gain - gso_par.added_loss
        - ctx.fspl_dB - rain_att
    )

    if tx.model == "CONSTANT_PFD_AT_GND":
        # NOTE: sum power before subtracting rain attenuation
        # because of numpy shape. Should give same result
        interf_pow_per_sat = (
            tx.pfd_at_ref_bandwidth_dBW_m2
            + 10 * np.log10(ctx.lmbda**2 / (4 * np.pi))
            + ant_rx_gain - rain_att
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

    return {"c": carrier_pow_dl_per_mhz, "cn": cn, "cni": cni,
            "i": interf_pow, "epfd": epfd, "gso_rain_att": rain_att}


def run_simulation(
    par: ParametersNGSO2GSO,
    results_writer: ResultsWriter,
    par_file: Path = None,
    start_from_drop=0,
):
    orbit_models = [
        OrbitModel(
            Nsp=p.sats_per_plane, Np=p.n_planes,
            phasing=p.phasing_deg, long_asc=p.long_asc_deg,
            omega=p.omega_deg, delta=p.inclination_deg,
            hp=p.perigee_alt_km, ha=p.apogee_alt_km,
            Mo=p.initial_mean_anomaly,
            # IGNORE THIS
            model_time_as_random_variable=False,
            t_min=0, t_max=-1,
        )
        for p in par.ngso.orbits
    ]
    total_sats = sum(p.n_planes * p.sats_per_plane for p in par.ngso.orbits)

    gso_contexts = [_build_gso_link_context(gso_par) for gso_par in par.gso_links]
    es_groups = _group_contexts_by_earth_station(gso_contexts)

    timeline = np.arange(par.min_t_s, par.max_t_s, par.delta_t_s)
    n_steps = len(timeline)

    seed_seq = np.random.SeedSequence(par.seed)
    child_seed_seqs = seed_seq.spawn(n_steps)

    p_rain_all = np.array([
        np.random.default_rng(s).uniform(0, 100)
        for s in child_seed_seqs
    ])  # cheap — just n_steps floats
    generic_rng = np.random.default_rng(par.seed)
    # orbital_rng = np.random.RandomState(par.seed)

    # Tune this to your available RAM. At 3232 sats:
    #   chunk=1000  → ~75MB for orbit positions
    #   chunk=10000 → ~750MB
    # par.batch_size
    orig_steps = np.array(range(0, n_steps, par.batch_size))
    actual_steps = orig_steps[orig_steps >= start_from_drop]

    # NOTE: this is cause I'm assuming that we're not missing in-between drops
    # later on
    assert actual_steps[0] == start_from_drop

    start_at_chunk = len(orig_steps) - len(actual_steps)

    already_plotted = False

    for chunk_start in tqdm(
        actual_steps, desc="chunks",
        total=len(orig_steps),
        initial=start_at_chunk
    ):
        chunk_end = min(chunk_start + par.batch_size, n_steps)
        chunk_timeline = timeline[chunk_start:chunk_end]
        chunk_p_rain = p_rain_all[chunk_start:chunk_end]   # (chunk,)

        # --- Batch orbit propagation for this chunk ---
        all_orbit_positions = [
            # testing random positions
            # o.get_orbit_positions_random(orbital_rng, len(chunk_timeline))
            o.get_orbit_positions_time_instant(chunk_timeline)
            for o in orbit_models
        ]
        # (chunk, total_sats)
        ngso_x_ecef = np.vstack([pos['sx'] for pos in all_orbit_positions]) * 1e3
        ngso_y_ecef = np.vstack([pos['sy'] for pos in all_orbit_positions]) * 1e3
        ngso_z_ecef = np.vstack([pos['sz'] for pos in all_orbit_positions]) * 1e3

        # --- Batch rain attenuation for this chunk ---
        rain_att_chunk: dict[str, np.ndarray] = {
            ctx.label: ctx.rain_inv_ccdf(chunk_p_rain).value[0]
            for ctx in gso_contexts
        }
        # --- Step loop within chunk ---
        for i in tqdm(range(len(chunk_timeline)), desc="steps", leave=False):
            for es_key, ctx_group in es_groups.items():
                ref_ctx = ctx_group[0]

                nx, ny, nz = ref_ctx.global_coord_sys.ecef2enu(
                    ngso_x_ecef[:, i], ngso_y_ecef[:, i], ngso_z_ecef[:, i]
                )
                ngso_geom = SimulatorGeometry(total_sats)
                ngso_geom.set_global_coords(nx, ny, nz)

                off_axis = ref_ctx.es_geom.get_off_axis_angle(ngso_geom)[0]
                elevation = ref_ctx.es_geom.get_local_elevation(ngso_geom)[0]
                off_axis = off_axis[elevation > par.minimum_elevation]
                elevation = elevation[elevation > par.minimum_elevation]

                if par.ngso.gso_protection_avoidance_angle is not None:
                    elevation = elevation[off_axis > par.ngso.gso_protection_avoidance_angle]
                    off_axis = off_axis[off_axis > par.ngso.gso_protection_avoidance_angle]

                if par.selection_strategy == "RAND":
                    n = min(par.ngso.n_co_channel, len(off_axis))
                    selected = generic_rng.choice(
                        len(off_axis),
                        size=n,
                        replace=False
                    )
                elif par.selection_strategy == "MAX_ELEV":
                    selected = np.argsort(-elevation)[:par.ngso.n_co_channel]
                elif par.selection_strategy == "WC":
                    selected = np.argsort(off_axis)[:par.ngso.n_co_channel]

                selected_off_axis = off_axis[selected]

                for ctx in ctx_group:
                    ant_rx_gain = es_ant_gain_1428(
                        selected_off_axis,
                        ctx.par.peak_rx_antenna_gain,
                        ctx.es_rx_ant_d_lmbda,
                    )
                    metrics = _compute_gso_link_metrics(
                        ctx, off_axis, selected, ant_rx_gain,
                        rain_att_chunk[ctx.label][i], par
                    )
                    results_writer.add_results(
                        metrics, f"gso_per_iteration{STR_SEPARATOR}{ctx.label}"
                    )
                    equivalent_rx_gain = 10*np.log10(
                        np.sum(10**(ant_rx_gain/10))
                    )
                    epfd_from_pfd = (
                        par.ngso.tx_model.pfd_at_ref_bandwidth_dBW_m2
                        + equivalent_rx_gain - ctx.par.peak_rx_antenna_gain
                    )
                    # results_writer.add_results(
                    #     {
                    #         "off_axis": selected_off_axis,
                    #         "ant_rx_gain": ant_rx_gain,
                    #     }, f"gso_per_ngso_per_iteration{STR_SEPARATOR}{ctx.label}"
                    # )
                    # results_writer.add_results(
                    #     {
                    #         "equivalent_rx_gain": equivalent_rx_gain,
                    #         "epfd_from_pfd": epfd_from_pfd,
                    #     }, f"extra_info{STR_SEPARATOR}{ctx.label}"
                    # )

                    if not already_plotted and DEBUG:
                        already_plotted = True
                    if not already_plotted and DEBUG:
                        already_plotted = True
                        plot_scenario(
                            ctx.global_coord_sys,
                            ctx.es_geom, ctx.ss_geom, ngso_geom
                        ).show()

        results_writer.flush_results()

    results_writer.flush_results()
    results_writer.close()


def create_results_writer(
    param_file: Path, par: ParametersNGSO2GSO
):
    # return MockResultsWriter()

    params_text = param_file.read_text()

    if RESULTS_JSON_RESUME_PATH.exists():
        results_json_resume = json.loads(
            RESULTS_JSON_RESUME_PATH.read_text()
        )
    else:
        results_json_resume = {}

    latest_result_dir = None
    if par.scenario_name in results_json_resume:
        latest_result_dir = Path(results_json_resume[par.scenario_name]['latest_res_dir'])

    if latest_result_dir is not None and latest_result_dir.exists():
        latest_file = list(latest_result_dir.glob("*.yaml"))[0]
        latest_file_params_text = latest_file.read_text()
    else:
        latest_file = None
        latest_file_params_text = None

    results_writer = None
    drops_ran = 0
    drops_start = 0

    if latest_file_params_text == params_text:
        csvs_at_dir = list(latest_result_dir.glob("gso_per_iteration*.csv"))
        if len(csvs_at_dir) != 0:
            per_drop_csv = csvs_at_dir[0]
            drops_ran = len(per_drop_csv.read_text().split('\n')) - 2
            drops_should_run = (par.max_t_s - par.min_t_s) / par.delta_t_s

            if drops_ran != drops_should_run:
                dt_str = str(latest_result_dir.name)
                timestamp = "_".join(dt_str.split("_")[-2:])
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                print(
                    "It seems there already a previous simulation "
                    "with the exact same parameters, started on \n"
                    f"{dt.year}-{dt.month}-{dt.day} {dt.hour}:{dt.minute}:{dt.second}"
                )
                res = None
                while res not in ["Y", "N"]:
                    res = input("Do you wish to continue the previous simulation? [Y/N]")
                    res = res.upper()

                if res == "Y":
                    yyyyyyyyyyy = drops_ran
                    results_writer = ResultsWriter(
                        f"{RESULTS_DIR}/", param_file, par.scenario_name,
                        timestamp
                    )

    if results_writer is None:
        results_writer = ResultsWriter(
            f"{RESULTS_DIR}/", param_file, par.scenario_name,
        )

    results_json_resume.setdefault(
        par.scenario_name, {}
    )['latest_res_dir'] = str(results_writer.directory)

    # RESULTS_JSON_RESUME_PATH.write_text(json.dumps(
    #     results_json_resume, sort_keys=True, indent=4
    # ))

    return results_writer, drops_start


def main():
    parser = argparse.ArgumentParser(
        description="SHARC - Radio Sharing and Compatiblity Monte Carlo Simulator"
    )
    parser.add_argument("-p", "--param-file", help="Path to parameter file")
    # parser.add_argument(
    #     "-ni", "--non-interactive",
    #     help="By default, this cmd may prompt you for answers"
    # )
    args = parser.parse_args()

    if Path(args.param_file).is_absolute():
        param_file = Path(args.param_file)
    else:
        param_file = Path(".") / args.param_file

    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(param_file)
    par.validate("ngso2gso")

    results_writer, drops_ran = create_results_writer(param_file, par)

    print(f"Results will be saved on {results_writer.directory}")

    run_simulation(
        par, results_writer, param_file,
        start_from_drop=drops_ran
    )


if __name__ == "__main__":
    main()
