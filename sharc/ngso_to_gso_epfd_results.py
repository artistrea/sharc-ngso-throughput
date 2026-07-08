"""
plot_epfd.py
------------
Plot the EPFD CCDF for each GSO link, re-referenced to a chosen standard
antenna, overlaid with Article 22 / RR Appendix 30B limit curves.

Produces two figures per run:
  - epfd_ccdf_lower_band_<diam>m.png  (17.3-17.7 / 17.8-18.6 GHz limit)
  - epfd_ccdf_upper_band_<diam>m.png  (19.7-20.2 GHz limit)

Usage
-----
    python plot_epfd.py -r <results_dir> --antenna-diameter 0.9
    python plot_epfd.py -r <results_dir> --antenna-diameter 1.0

Re-referencing formula
----------------------
    epfd_ref = I - G_rx + 10·log10(4π/λ_ref²)

where:
    I       = stored interference power  [dBW]
    G_rx    = peak gain of the actual receiver antenna  [dBi]
    G_ref   = peak gain of the reference antenna at ref_freq  [dBi]
    λ_ref   = wavelength at the band centre used for the limit
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from pathlib import Path
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR
)

RESULTS_DIR = "./results-ngso"
RESULTS_JSON_RESUME_PATH = Path(f"{RESULTS_DIR}/results.json")

# ---------------------------------------------------------------------------
# Article 22 / RR Appendix 30B limits  —  B = 1 MHz
# Each entry: (epfd dBW/m², % time NOT exceeded)
# Duplicate % values create vertical steps in the CCDF limit staircase.
# ---------------------------------------------------------------------------
BAND_LIMITS = {
    "lower": {
        "label":        "17.3–17.7 / 17.8–18.6 GHz",
        "center_GHz":   (17.3 + 18.6) / 2,   # 17.95 GHz
        "ref_bw_Hz":    1e6,
        "points": [
            (-161.4,   0.0   ),
            (-161.4,  90.0   ),   # vertical step at 90 %
            (-158.5,  99.0   ),
            (-153.0,  99.714 ),
            (-150.0,  99.971 ),
            (-150.0, 100.0   ),
        ],
    },
    "upper": {
        "label":        "19.7–20.2 GHz",
        "center_GHz":   (19.7 + 20.2) / 2,   # 19.95 GHz
        "ref_bw_Hz":    1e6,
        "points": [
            (-176.4,   0.0   ),
            (-167.4,  91.0   ),
            (-156.4,  99.8   ),
            (-154.6,  99.8   ),   # vertical step at 99.8 %
            (-151.,  99.943 ),
            (-146.,  99.943 ),   # vertical step at 99.943 %
            (-140.,  99.997 ),
            (-140., 100.0   ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Reference antenna peak gain  (ITU-R S.1428-1)
# ---------------------------------------------------------------------------

def reference_antenna_peak_gain(diameter_m: float, freq_GHz: float) -> float:
    """
    Peak gain of a reference antenna per ITU-R S.1428-1.

    D/λ > 100  →  G_max = 20·log10(D/λ) + 8.4
    25 ≤ D/λ ≤ 100  →  G_max = 20·log10(D/λ) + 7.7
    """
    lmbda = 3e8 / (freq_GHz * 1e9)
    d_over_lmbda = diameter_m / lmbda
    if d_over_lmbda > 100:
        return 20 * np.log10(d_over_lmbda) + 8.4
    elif d_over_lmbda >= 25:
        return 20 * np.log10(d_over_lmbda) + 7.7
    else:
        raise ValueError(
            f"D/λ = {d_over_lmbda:.1f} < 25 for {diameter_m} m at "
            f"{freq_GHz} GHz — S.1428-1 pattern not defined."
        )


# ---------------------------------------------------------------------------
# EPFD re-referencing
# ---------------------------------------------------------------------------

def reref_epfd(
    i_dBW: np.ndarray,
    g_rx_dBi: float,
    ref_freq_GHz: float,
) -> np.ndarray:
    """
    Re-reference interference power I to a standard antenna EPFD.

        epfd_ref = I - G_rx + G_ref + 10·log10(4π/λ_ref²)

    Parameters
    ----------
    i_dBW         : interference power time series  [dBW]
    g_rx_dBi      : peak gain of the actual receiver antenna  [dBi]
    ref_freq_GHz  : band centre frequency for λ and G_ref  [GHz]
    """
    lmbda_ref = 3e8 / (ref_freq_GHz * 1e9)
    pfd_factor = 10 * np.log10(4 * np.pi / lmbda_ref**2)

    return i_dBW - g_rx_dBi + pfd_factor


# ---------------------------------------------------------------------------
# Empirical CCDF  (no interpolation)
# ---------------------------------------------------------------------------

def empirical_ccdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    P(X >= x) as a percentage.
    Exceedance = (n - rank + 1) / n * 100  so the max maps to 100/n not 0.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    exceedance_pct = np.arange(n, 0, -1) / n * 100.0
    return sorted_vals, exceedance_pct


# ---------------------------------------------------------------------------
# Limit curve helper
# ---------------------------------------------------------------------------

def limit_curve(band_key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the (epfd, % not exceeded) CDF limit into CCDF exceedance form.
    Returns (epfd_dBW_m2, exceedance_pct).
    """
    pts = np.array(BAND_LIMITS[band_key]["points"])
    epfd_vals      = pts[:, 0]
    pct_not_exc    = pts[:, 1]
    exceedance_pct = 100.0 - pct_not_exc
    return epfd_vals, exceedance_pct


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_link_data(
    results_dir: Path, par: ParametersNGSO2GSO
) -> dict[str, pd.DataFrame]:
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


# ---------------------------------------------------------------------------
# Single-band plot
# ---------------------------------------------------------------------------

def _plot_epfd_band(
    band_key: str,
    link_data: dict[str, pd.DataFrame],
    gso_par_map: dict,
    plots_dir: Path,
    colours: dict[str, tuple],
):
    band = BAND_LIMITS[band_key]
    if band_key == "upper":
        ref_diameter_m = 0.9
    elif band_key == "lower":
        ref_diameter_m = 1.
    else:
        raise ValueError("Ai nao, neh")

    ref_freq_GHz = band["center_GHz"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Article 22 limit ---
    lim_epfd, lim_exc = limit_curve(band_key)
    ax.semilogy(
        lim_epfd, np.maximum(lim_exc, 1e-4),   # clip 0 % for log axis
        color="black", linewidth=2.0, linestyle="--",
        label=f"Art. 22 limit  ({band['label']},  1 MHz)",
        zorder=10,
    )

    all_epfd_vals = []

    for label, df in link_data.items():
        if "i" not in df.columns:
            print(f"[WARN] Column 'i' not in {label}, skipping.")
            continue

        gso_par  = gso_par_map[label]
        epfd_ref = reref_epfd(
            i_dBW          = df["i"].to_numpy(),
            g_rx_dBi       = gso_par.peak_rx_antenna_gain,
            ref_freq_GHz   = ref_freq_GHz,
        )
        all_epfd_vals.append(epfd_ref)

        x, exc_pct = empirical_ccdf(epfd_ref)
        ax.semilogy(
            x, exc_pct,
            color=colours[label], label=label, linewidth=1.2,
        )

    ax.set_xlabel("EPFD  (dBW/m²)")
    ax.set_ylabel("% time EPFD is exceeded  [P(X ≥ x)]")
    ax.set_title(
        f"EPFD CCDF vs Article 22 limit  —  {band['label']}\n"
        f"Reference antenna: {ref_diameter_m} m  |  BW: 1 MHz"
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=7, loc="upper right")

    # if all_epfd_vals:
    #     combined = np.concatenate(all_epfd_vals)
    #     x_min = min(combined.min(), lim_epfd.min()) - 5
    #     x_max = max(combined.max(), lim_epfd.max()) + 5
    #     ax.set_xlim(x_min, x_max)
    ax.set_xlim(-200., -130.)
    ax.set_ylim(1e-3, 1e2)

    fig.tight_layout()

    diam_str = f"{ref_diameter_m:.2f}m".replace(".", "p")
    out_path = plots_dir / f"epfd_ccdf_{band_key}_band_{diam_str}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def plot_epfd(
    link_data: dict[str, pd.DataFrame],
    par: ParametersNGSO2GSO,
    results_dir: Path,
):
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    labels  = list(link_data.keys())
    cmap    = plt.get_cmap("tab10")
    colours = {label: cmap(i % 10) for i, label in enumerate(labels)}

    gso_par_map = {gso.full_name: gso for gso in par.gso_links}

    for band_key in ("lower", "upper"):
        _plot_epfd_band(
            band_key       = band_key,
            link_data      = link_data,
            gso_par_map    = gso_par_map,
            plots_dir      = plots_dir,
            colours        = colours,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot CCDF/CDF results from NGSO-to-GSO simulation"
    )
    parser.add_argument("-rd", "--results-dir", help="Path to results dir")
    parser.add_argument(
        "-rn", "--results-name", help="Results name (plots latest result)"
    )
    args = parser.parse_args()

    if args.results_dir is None and args.results_name is None:
        raise ValueError("You must pass either results name or dir")
    if args.results_dir is not None and args.results_name is not None:
        raise ValueError("You may not pass both results name and dir")

    if args.results_name is not None:
        if RESULTS_JSON_RESUME_PATH.exists():
            results_json_resume = json.loads(
                RESULTS_JSON_RESUME_PATH.read_text()
            )
        else:
            raise ValueError("Could not find results resume json")
        if args.results_name not in results_json_resume:
            raise ValueError(
                f"Could not find {args.results_name} in the results resume json"
            )
        results_dir = results_json_resume[args.results_name]['latest_res_dir']
        results_dir = (
            Path(results_dir)
            if Path(results_dir).is_absolute()
            else Path(".") / results_dir
        )
    else:
        results_dir = (
            Path(args.results_dir)
            if Path(args.results_dir).is_absolute()
            else Path(".") / args.results_dir
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

    plot_epfd(link_data, par, results_dir)


if __name__ == "__main__":
    main()
