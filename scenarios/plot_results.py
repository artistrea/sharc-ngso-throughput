import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR
)

SHARC_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_JSON_RESUME_PATH = RESULTS_DIR / "results.json"

PROTECTION_PARAMS = {
    "RAND": [
        (2, 2), (3, 2), (4, 2),
        (2, 4), (3, 4), (4, 4),
        (2, 6), (3, 6), (4, 6),
        (2, 8), (3, 8), (4, 8),
    ],
    "WC": [
        (5, 2), (6, 2), (7, 2),
        (5, 4), (6, 4), (7, 4),
        (5, 6), (6, 6), (7, 6),
        (5, 8), (6, 8), (7, 8),
    ],
    "MAX_ELEV": [
        (1, 2), (2, 2), (3, 2),
        (1, 4), (2, 4), (3, 4),
        (1, 6), (2, 6), (3, 6),
        (1, 8), (2, 8), (3, 8),
    ],
}


def load_link_data(
    results_dir: Path, par: ParametersNGSO2GSO
) -> dict[str, pd.DataFrame]:
    """Load all per-link CSVs. Returns {link_label: DataFrame}."""
    link_data = {}
    for gso_par in par.gso_links:
        label = gso_par.full_name
        filename = f"gso_per_iteration{STR_SEPARATOR}{label}.parquet"
        path = results_dir / filename
        if not path.exists():
            print(f"[WARN] Missing results file: {path}")
            continue
        link_data[label] = pd.read_parquet(path)
    return link_data


def spectral_efficiency(gamma_db: np.ndarray) -> np.ndarray:
    """
    Piecewise spectral efficiency η(γ) in bps/Hz, per Equation 3.

        η = 0                                       γ < -8.9
        η = 0.376643 + 0.030337·γ²        -8.9 ≤ γ < -2.5
        η = 0.5933 + 0.1415·γ + 0.0096·γ² -2.5 ≤ γ < 0
        η = 0.5933 + 0.1388·γ + 0.003·γ²     0 ≤ γ < 25.02
        η = 5.944                               γ ≥ 25.02

    Parameters
    ----------
    gamma_db : C/N or C/(N+I) in dB, any shape

    Returns
    -------
    eta : spectral efficiency in bps/Hz, same shape as gamma_db
    """
    g = np.asarray(gamma_db, dtype=float)
    eta = np.empty_like(g)

    eta[g < -8.9] = 0.0

    m = (-8.9 <= g) & (g < -2.5)
    eta[m] = 0.376643 + 0.030337 * g[m]**2

    m = (-2.5 <= g) & (g < 0.0)
    eta[m] = 0.5933 + 0.1415 * g[m] + 0.0096 * g[m]**2

    m = (0.0 <= g) & (g < 25.02)
    eta[m] = 0.5933 + 0.1388 * g[m] + 0.003 * g[m]**2

    eta[g >= 25.02] = 5.944

    return eta


def main():
    results_json_resume = {}

    last_tstamp = None
    maybe_included_dirs = reversed(sorted(RESULTS_DIR.iterdir()))
    for d in reversed(sorted(RESULTS_DIR.iterdir())):
        try:
            timestamp: datetime.datetime = datetime.strptime(str(d.name), "%Y%m%d_%H%M%S")
        except Exception:
            continue

        if last_tstamp is not None:
            if (last_tstamp - timestamp) > timedelta(seconds=10):
                # assume they're too old of a run
                break
        last_tstamp = timestamp

        for res_dir in d.iterdir():
            results_json_resume[res_dir.name] = {
                "latest_res_dir": "./" + str(res_dir.relative_to(SHARC_ROOT))
            }

    data = {}
    for k, v in results_json_resume.items():
        results_dir = v["latest_res_dir"]
        results_dir = (
            Path(results_dir)
            if Path(results_dir).is_absolute()
            else Path(".") / results_dir
        )
        possible_param_files = list(results_dir.glob("*.yaml"))
        if len(possible_param_files) == 0:
            raise ValueError("No .yaml param file found in results dir")
        if len(possible_param_files) > 1:
            raise ValueError(f"Ambiguous param files: {possible_param_files}")

        par = ParametersNGSO2GSO()
        par.load_parameters_from_file(possible_param_files[0])
        par.validate("ngso2gso")

        link_data = load_link_data(results_dir, par)
        if not link_data:
            raise ValueError("No link data loaded — check results dir and link labels.")

        max_rel = 0.
        for i, (label, df) in enumerate(link_data.items()):
            if "cn" not in df.columns or "cni" not in df.columns:
                print(f"[WARN] cn/cni missing for {label}, skipping throughput.")
                continue

            eta_cn = spectral_efficiency(df["cn"].to_numpy())
            eta_cni = spectral_efficiency(df["cni"].to_numpy())

            ser = float(np.mean(eta_cn))
            seri = float(np.mean(eta_cni))

            rel = (
                (ser - seri) / ser * 100.0
                if (np.isfinite(ser) and ser > 0)
                else float("nan")
            )
            if rel > max_rel:
                max_rel = rel

        data[k] = max_rel

    selec_grouping = {}
    for selection_strategy in ["RAND", "WC", "MAX_ELEV"]:
        arc_avoid_grouping = {}
        for arc_avoid, n_co in PROTECTION_PARAMS[selection_strategy]:
            scenario_name = f"{selection_strategy.lower()}_{arc_avoid}arc_avoid_{n_co}Nco"
            if scenario_name in data:
                arc_avoid_grouping.setdefault(arc_avoid, {})[n_co] = data[scenario_name]
        selec_grouping[selection_strategy] = arc_avoid_grouping
    print(selec_grouping)

    # TODO:
    # one plot per scenario
    # one line per arc avoidance angle
    # n_co sweep (x axis)
    # maximum throughput degradation [%] (y axis)
    import matplotlib.pyplot as plt

    for selection_strategy, arc_avoid_grouping in selec_grouping.items():
        if not arc_avoid_grouping:
            continue

        plt.figure()

        for arc_avoid, n_co_grouping in sorted(arc_avoid_grouping.items()):
            n_co_values = sorted(n_co_grouping)
            degradation_values = [
                n_co_grouping[n_co] for n_co in n_co_values
            ]

            plt.plot(
                n_co_values,
                degradation_values,
                marker="o",
                label=f"{arc_avoid}°",
            )

        plt.xlabel("$N_{co}$")
        plt.ylabel("Maximum throughput degradation [%]")
        plt.title(f"{selection_strategy} satellite selection")
        plt.grid(True)
        plt.legend(title="Arc avoidance")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
