import json
import pandas as pd
import numpy as np
from pathlib import Path
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_JSON_RESUME_PATH = RESULTS_DIR / "results.json"


def load_link_data(
    results_dir: Path, par: ParametersNGSO2GSO
) -> dict[str, pd.DataFrame]:
    """Load all per-link CSVs. Returns {link_label: DataFrame}."""
    link_data = {}
    for gso_par in par.gso_links:
        label = gso_par.full_name
        filename = f"gso_per_iteration{STR_SEPARATOR}{label}.csv"
        path = results_dir / filename
        if not path.exists():
            print(f"[WARN] Missing results file: {path}")
            continue
        link_data[label] = pd.read_csv(path)
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

            rel  = (
                (ser - seri) / ser * 100.0
                if (np.isfinite(ser) and ser > 0)
                else float("nan")
            )
            if rel > max_rel:
                max_rel = rel

        data[k] = max_rel

    print(data)


if __name__ == "__main__":
    main()
