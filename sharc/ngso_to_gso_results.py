import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR
)

# Columns to plot CCDFs for, with axis labels
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


def empirical_ccdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (sorted_values, exceedance_probability) for an empirical CCDF.
    No interpolation — uses only the observed sample values.
    P(X >= x) estimated as (n - rank + 1) / n so the largest value maps to
    1/n rather than 0.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    exceedance = np.arange(n, 0, -1) / n
    return sorted_vals, exceedance


def empirical_cdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (sorted_values, cumulative_probability) for an empirical CDF.
    P(X <= x) estimated as rank / n.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    cumulative = np.arange(1, n + 1) / n
    return sorted_vals, cumulative


def outage_at_threshold(values: np.ndarray, threshold: float) -> float:
    """
    Percentage of samples strictly below threshold — i.e. 100 * P(X < threshold).
    Empirical only: count samples, no interpolation.
    """
    return float(np.mean(values < threshold)) * 100.0


def load_link_data(results_dir: Path, par: ParametersNGSO2GSO) -> dict[str, pd.DataFrame]:
    """
    Load all per-link CSVs from the results directory.
    Returns {link_label: DataFrame}.
    """
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


def _plot_distribution(
    link_data: dict[str, pd.DataFrame],
    plots_dir: Path,
    colours: dict[str, tuple],
    mode: str,  # "ccdf" or "cdf"
) -> dict[str, dict[str, float]]:
    """
    Plot CCDF or CDF for every metric, one figure per metric.
    Returns outage_table only when mode=="ccdf" (CCDFs carry the threshold markers).
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
                        linestyle="--", zorder=0
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
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=7, loc="upper right" if mode == "ccdf" else "lower right")
        fig.tight_layout()

        out_path = plots_dir / f"{mode}_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    return outage_table


def plot_distributions(
    link_data: dict[str, pd.DataFrame],
    results_dir: Path,
) -> dict[str, dict[str, float]]:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    labels = list(link_data.keys())
    cmap = plt.get_cmap("tab10")
    colours = {label: cmap(i % 10) for i, label in enumerate(labels)}

    outage_table = _plot_distribution(link_data, plots_dir, colours, mode="ccdf")
    _plot_distribution(link_data, plots_dir, colours, mode="cdf")

    return outage_table


def format_and_save_outage_table(
    outage_table: dict[str, dict[str, float]],
    results_dir: Path,
):
    """
    Print and save a summary table with:
      - Pout(C/N < 0 dB)        [%]   — rain-only unavailability
      - Pout(C/(N+I) < 0 dB)    [%]   — unavailability with NGSO interference
      - relative increase        [%]   — (Pout_cni - Pout_cn) / Pout_cn * 100
    """
    cn_key  = "Pout_cn_lt_0.0dB"
    cni_key = "Pout_cni_lt_0.0dB"

    col_link  = 35
    col_cn    = 22
    col_cni   = 24
    col_rel   = 22
    total_w   = col_link + col_cn + col_cni + col_rel + 6

    header = (
        f"{'Link':<{col_link}}  "
        f"{'Pout rain-only (%)':>{col_cn}}  "
        f"{'Pout w/ NGSO (%)':>{col_cni}}  "
        f"{'Rel. increase (%)':>{col_rel}}"
    )
    separator = "-" * total_w

    rows = []
    for label, stats in outage_table.items():
        p_cn  = stats.get(cn_key,  float("nan"))
        p_cni = stats.get(cni_key, float("nan"))

        if np.isfinite(p_cn) and p_cn > 0:
            rel_increase = (p_cni - p_cn) / p_cn * 100.0
        else:
            rel_increase = float("nan")

        rows.append((label, p_cn, p_cni, rel_increase))

    lines = [header, separator]
    for label, p_cn, p_cni, rel in rows:
        lines.append(
            f"{label:<{col_link}}  "
            f"{p_cn:>{col_cn}.4f}  "
            f"{p_cni:>{col_cni}.4f}  "
            f"{rel:>{col_rel}.2f}"
        )

    table_str = "\n".join(lines)
    print()
    print(table_str)
    print()

    out_path = results_dir / "outage_summary.txt"
    out_path.write_text(table_str + "\n")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CCDF results from NGSO-to-GSO simulation"
    )
    parser.add_argument("-r", "--results-dir", help="Path to results dir")
    args = parser.parse_args()

    results_dir = (
        Path(args.results_dir)
        if Path(args.results_dir).is_absolute()
        else Path(".") / args.results_dir
    )

    possible_param_files = list(results_dir.glob("*.yaml"))
    if len(possible_param_files) == 0:
        raise ValueError("No .yaml param file found in results dir")
    if len(possible_param_files) > 1:
        raise ValueError(
            f"Ambiguous param files: {possible_param_files}"
        )

    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(possible_param_files[0])
    par.validate("ngso2gso")

    link_data = load_link_data(results_dir, par)
    if not link_data:
        raise ValueError("No link data loaded — check results dir and link labels.")

    outage_table = plot_distributions(link_data, results_dir)
    format_and_save_outage_table(outage_table, results_dir)


if __name__ == "__main__":
    main()
