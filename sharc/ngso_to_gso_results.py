import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR
)

RESULTS_DIR = "./results-ngso"
RESULTS_JSON_RESUME_PATH = Path(f"{RESULTS_DIR}/results.json")

# Columns to plot CCDFs/CDFs for, with axis labels
METRICS = {
    "c":           "C  (dBW/MHz)",
    "cn":          "C/N  (dB)",
    "cni":         "C/(N+I)  (dB)",
    "i":           "I  (dBW)",
    "epfd":        "EPFD  (dBW/m²)",
    "gso_rain_att":"Rain attenuation  (dB)",
}

# Thresholds for outage probability markers
THRESHOLD_0DB = {
    "cn":  0.0,
    "cni": 0.0,
}


# ---------------------------------------------------------------------------
# Spectral efficiency
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Empirical distributions
# ---------------------------------------------------------------------------

def empirical_ccdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    P(X >= x): exceedance = (n - rank + 1) / n  so the max maps to 1/n not 0.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    exceedance = np.arange(n, 0, -1) / n
    return sorted_vals, exceedance


def empirical_cdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    P(X <= x): cumulative = rank / n.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    cumulative = np.arange(1, n + 1) / n
    return sorted_vals, cumulative


def outage_at_threshold(values: np.ndarray, threshold: float) -> float:
    """100 * P(X < threshold) — empirical, no interpolation."""
    return float(np.mean(values < threshold)) * 100.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distribution plots (CCDF / CDF) for raw metrics
# ---------------------------------------------------------------------------

def _plot_distribution(
    link_data: dict[str, pd.DataFrame],
    plots_dir: Path,
    colours: dict[str, tuple],
    mode: str,                          # "ccdf" or "cdf"
) -> dict[str, dict[str, float]]:
    """
    One figure per metric, all GSO links overlaid.
    Returns outage_table (populated only when mode=="ccdf").
    """
    outage_table: dict[str, dict[str, float]] = {
        label: {} for label in link_data
    }

    for metric, axis_label in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 5))

        for label, df in link_data.items():
            if metric not in df.columns:
                print(f"[WARN] Column '{metric}' not in {label}, skipping.")
                continue

            values = df[metric].to_numpy()

            if mode == "ccdf":
                x, p = empirical_ccdf(values)
                ylabel = "P(X ≥ x)"
            else:
                x, p = empirical_cdf(values)
                ylabel = "P(X ≤ x)"

            ax.semilogy(x, p, color=colours[label], label=label, linewidth=1.2)

            # Threshold markers on CCDF only
            if mode == "ccdf" and metric in THRESHOLD_0DB:
                threshold = THRESHOLD_0DB[metric]
                p_out = outage_at_threshold(values, threshold)
                outage_table[label][f"Pout_{metric}_lt_{threshold}dB"] = p_out

                if p_out > 0:
                    ax.axvline(
                        threshold, color="grey", linewidth=0.8,
                        linestyle="--", zorder=0,
                    )
                    above = p[x >= threshold]
                    p_at_threshold = above[0] if len(above) else np.nan
                    if np.isfinite(p_at_threshold):
                        ax.plot(
                            threshold, p_at_threshold,
                            marker="o", markersize=5,
                            color=colours[label], zorder=5,
                        )

        ax.set_xlabel(axis_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{mode.upper()} — {axis_label}")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.10f"))
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(
            fontsize=7,
            loc="upper right" if mode == "ccdf" else "lower right",
        )
        fig.tight_layout()

        out_path = plots_dir / f"{mode}_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    return outage_table


# ---------------------------------------------------------------------------
# Spectral efficiency / throughput CDF plots
# ---------------------------------------------------------------------------

def plot_throughput_cdfs(
    link_data: dict[str, pd.DataFrame],
    plots_dir: Path,
    colours: dict[str, tuple],
) -> dict[str, dict[str, float]]:
    """
    For each GSO link compute η(C/N) and η(C/(N+I)), plot their CDFs on one
    shared figure, and return the time-averaged SER and SERI per link.

    Returns
    -------
    ser_table : {label: {"SER": float, "SERI": float}}
    """
    ser_table: dict[str, dict[str, float]] = {}

    fig_cn,  ax_cn  = plt.subplots(figsize=(9, 5))
    fig_cni, ax_cni = plt.subplots(figsize=(9, 5))

    for label, df in link_data.items():
        colour = colours[label]

        if "cn" not in df.columns or "cni" not in df.columns:
            print(f"[WARN] cn/cni missing for {label}, skipping throughput.")
            continue

        eta_cn  = spectral_efficiency(df["cn"].to_numpy())
        eta_cni = spectral_efficiency(df["cni"].to_numpy())

        ser  = float(np.mean(eta_cn))
        seri = float(np.mean(eta_cni))
        ser_table[label] = {"SER": ser, "SERI": seri}

        # CDF of η(C/N)
        x_cn, p_cn = empirical_cdf(eta_cn)
        ax_cn.plot(x_cn, p_cn, color=colour, label=label, linewidth=1.2)

        # CDF of η(C/(N+I))
        x_cni, p_cni = empirical_cdf(eta_cni)
        ax_cni.plot(x_cni, p_cni, color=colour, label=label, linewidth=1.2)

    for ax, title, fname in [
        (ax_cn,  "CDF — Spectral efficiency η(C/N)  (bps/Hz)",
         "cdf_throughput_cn.png"),
        (ax_cni, "CDF — Spectral efficiency η(C/(N+I))  (bps/Hz)",
         "cdf_throughput_cni.png"),
    ]:
        ax.set_xlabel("Spectral efficiency η  (bps/Hz)")
        ax.set_ylabel("P(η ≤ x)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=7, loc="lower right")

    fig_cn.tight_layout()
    fig_cni.tight_layout()

    p_cn_out  = plots_dir / "cdf_throughput_cn.png"
    p_cni_out = plots_dir / "cdf_throughput_cni.png"
    fig_cn.savefig(p_cn_out,   dpi=150)
    fig_cni.savefig(p_cni_out, dpi=150)
    plt.close(fig_cn)
    plt.close(fig_cni)
    print(f"Saved: {p_cn_out}")
    print(f"Saved: {p_cni_out}")

    return ser_table


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _table_lines(header: str, separator: str, rows: list[str]) -> list[str]:
    return [header, separator] + rows


def format_and_save_outage_table(
    outage_table: dict[str, dict[str, float]],
    results_dir: Path,
):
    """
    Pout(C/N < 0 dB)  |  Pout(C/(N+I) < 0 dB)  |  relative increase
    All values in %.
    """
    cn_key  = "Pout_cn_lt_0.0dB"
    cni_key = "Pout_cni_lt_0.0dB"

    CL, CCN, CCNI, CREL = 35, 22, 24, 22

    header = (
        f"{'Link':<{CL}}  "
        f"{'Pout rain-only (%)':>{CCN}}  "
        f"{'Pout w/ NGSO (%)':>{CCNI}}  "
        f"{'Rel. increase (%)':>{CREL}}"
    )
    separator = "-" * (CL + CCN + CCNI + CREL + 6)

    rows = []
    for label, stats in outage_table.items():
        p_cn  = stats.get(cn_key,  float("nan"))
        p_cni = stats.get(cni_key, float("nan"))
        rel   = (
            (p_cni - p_cn) / p_cn * 100.0
            if (np.isfinite(p_cn) and p_cn > 0)
            else float("nan")
        )
        rows.append(
            f"{label:<{CL}}  {p_cn:>{CCN}.10f}  {p_cni:>{CCNI}.10f}  {rel:>{CREL}.10f}"
        )

    table_str = "\n".join(_table_lines(header, separator, rows))
    print("\n" + table_str + "\n")

    out_path = results_dir / "outage_summary.txt"
    out_path.write_text(table_str + "\n")
    print(f"Saved: {out_path}")


def format_and_save_ser_table(
    ser_table: dict[str, dict[str, float]],
    results_dir: Path,
):
    """
    SER (rain-only)  |  SERI (w/ NGSO)  |  (SER-SERI)/SER  |  relative reduction %

    The criterion from the standard is:
        (SER - SERI) / SER <= I_thr
    """
    CL, CSER, CSERI, CREL = 35, 20, 20, 26

    header = (
        f"{'Link':<{CL}}  "
        f"{'SER (bps/Hz)':>{CSER}}  "
        f"{'SERI (bps/Hz)':>{CSERI}}  "
        f"{'(SER-SERI)/SER (%)':>{CREL}}"
    )
    separator = "-" * (CL + CSER + CSERI + CREL + 6)

    rows = []
    for label, vals in ser_table.items():
        ser  = vals["SER"]
        seri = vals["SERI"]
        rel  = (
            (ser - seri) / ser * 100.0
            if (np.isfinite(ser) and ser > 0)
            else float("nan")
        )
        rows.append(
            f"{label:<{CL}}  {ser:>{CSER}.10f}  {seri:>{CSERI}.10f}  {rel:>{CREL}.10f}"
        )

    table_str = "\n".join(_table_lines(header, separator, rows))
    print("\n" + table_str + "\n")

    out_path = results_dir / "ser_summary.txt"
    out_path.write_text(table_str + "\n")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def plot_distributions(
    link_data: dict[str, pd.DataFrame],
    results_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Run all plots and return (outage_table, ser_table).
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    labels = list(link_data.keys())
    cmap = plt.get_cmap("tab10")
    colours = {label: cmap(i % 10) for i, label in enumerate(labels)}

    outage_table = _plot_distribution(link_data, plots_dir, colours, mode="ccdf")
    _plot_distribution(link_data, plots_dir, colours, mode="cdf")
    ser_table = plot_throughput_cdfs(link_data, plots_dir, colours)

    return outage_table, ser_table


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

    outage_table, ser_table = plot_distributions(link_data, results_dir)
    format_and_save_outage_table(outage_table, results_dir)
    format_and_save_ser_table(ser_table, results_dir)


if __name__ == "__main__":
    main()
