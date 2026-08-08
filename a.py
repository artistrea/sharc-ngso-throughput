"""
Scenario 3 throughput / spectral-efficiency plots.

Companion to the CCDF/CDF and EPFD-mask scripts. Focuses on the LONG-TERM
protection metric (time-weighted average spectral efficiency) for Scenario 3,
broken down by ground station / locality and by trial
(worst case / max elevation / random).

Unlike a hand-maintained list of links, this script discovers the GSO links
the SAME way the simulation does: it loads the .yaml parameter file saved
inside each results directory via ParametersNGSO2GSO and reads `par.gso_links`.
The per-link CSV label is `gso_par.full_name`, exactly the label used when the
results were written (gso_per_iteration{SEP}{full_name}.csv), so filenames match
by construction. Earth-station lat/lon, orbital slot and centre frequency are
pulled from the same parameters to label each station on the plots.

Spectral efficiency uses the ITU-R S.2131 Eq. (3) piecewise mapping (same as the
distribution script): SER = mean η(C/N) (rain only), SERI = mean η(C/(N+I))
(with NGSO). Long-term degradation is (SER - SERI) / SER.

Outputs (under <first-trial results dir>/plots-scenario3-<variant>/):
  bar_avg_throughput_by_station_<trial>.png
  bar_relative_decrease_by_station.png        (with 3% criterion line)
  cdf_throughput_by_station_<trial>.png
  box_throughput_by_station.png
  bar_avg_throughput_by_trial.png
  scenario3_throughput_summary.txt

Run:
  python plot_scenario3_throughput.py                 # variant "normal"
  # set SCENARIO3_VARIANT = "modified" for the modified run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, STR_SEPARATOR
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = "./results-ngso"
RESULTS_JSON_RESUME_PATH = Path(f"{RESULTS_DIR}/results.json")

# Long-term criterion (%) from the FCC Order / ITU Q-V framework.
CRITERION_PCT = 3.0

# Which Scenario-3 variant to plot.
#   "normal"   -> worst_case_scenario3 / max_elev_scenario3 / rand_scenario3
#   "modified" -> wc_scenario3_modified / max_elev_scenario3_modified / rand_scenario3_modified
# The slide notes want the modified run shown "as though normal scenario 3",
# so flip this to "modified" for that storyline.
SCENARIO3_VARIANT = "normal"

# res_names exactly as they appear in results.json. The worst-case prefix
# differs between variants (worst_case_ vs wc_), so both are spelled out.
_SCENARIO3_TRIALS_BY_VARIANT = {
    "normal": [
        {"res_name": "worst_case_scenario3", "label": "Worst Case"},
        {"res_name": "max_elev_scenario3",   "label": "Max. Elevation"},
        {"res_name": "rand_scenario3",       "label": "Random"},
    ],
    "modified": [
        {"res_name": "wc_scenario3_modified",       "label": "Worst Case"},
        {"res_name": "max_elev_scenario3_modified", "label": "Max. Elevation"},
        {"res_name": "rand_scenario3_modified",     "label": "Random"},
    ],
}
SCENARIO3_TRIALS = _SCENARIO3_TRIALS_BY_VARIANT[SCENARIO3_VARIANT]

OUT_SUBDIR = f"plots-scenario3-{SCENARIO3_VARIANT}"


# ---------------------------------------------------------------------------
# Spectral efficiency (ITU-R S.2131 Eq. 3) — identical to the CDF script
# ---------------------------------------------------------------------------

def spectral_efficiency(gamma_db: np.ndarray) -> np.ndarray:
    """Piecewise η(γ) in bps/Hz. γ is C/N or C/(N+I) in dB."""
    g = np.asarray(gamma_db, dtype=float)
    eta = np.empty_like(g)

    eta[g < -8.9] = 0.0

    m = (-8.9 <= g) & (g < -2.5)
    eta[m] = 0.376643 + 0.030337 * g[m] ** 2

    m = (-2.5 <= g) & (g < 0.0)
    eta[m] = 0.5933 + 0.1415 * g[m] + 0.0096 * g[m] ** 2

    m = (0.0 <= g) & (g < 25.02)
    eta[m] = 0.5933 + 0.1388 * g[m] + 0.003 * g[m] ** 2

    eta[g >= 25.02] = 5.944
    return eta


def empirical_cdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(X <= x): cumulative = rank / n."""
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    return sorted_vals, np.arange(1, n + 1) / n


# ---------------------------------------------------------------------------
# Results-resume + parameter-file loading (mirrors the simulation)
# ---------------------------------------------------------------------------

def results_name2dir(name: str) -> Path:
    if not RESULTS_JSON_RESUME_PATH.exists():
        raise ValueError(
            f"Could not find results resume json at {RESULTS_JSON_RESUME_PATH}"
        )
    resume = json.loads(RESULTS_JSON_RESUME_PATH.read_text())
    if name not in resume:
        raise ValueError(f"Could not find '{name}' in the results resume json")
    rd = resume[name]["latest_res_dir"]
    return Path(rd) if Path(rd).is_absolute() else Path(".") / rd


def load_params_from_results_dir(results_dir: Path) -> ParametersNGSO2GSO:
    """
    Load the .yaml saved inside a results directory, exactly like the sim.
    (The ResultsWriter copies the param file into the results dir on creation.)
    """
    yamls = list(results_dir.glob("*.yaml"))
    if len(yamls) == 0:
        raise ValueError(f"No .yaml param file found in {results_dir}")
    if len(yamls) > 1:
        raise ValueError(f"Ambiguous param files in {results_dir}: {yamls}")

    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(yamls[0])
    par.validate("ngso2gso")
    return par


def station_meta(gso_par) -> dict:
    """
    Human-readable metadata for one GSO link, from its parameters.
    Falls back gracefully if an attribute isn't present in this build.
    """
    es = getattr(gso_par, "earth_station", None)
    lat = getattr(es, "lat_deg", None) if es is not None else None
    lon = getattr(es, "lon_deg", None) if es is not None else None
    slot = getattr(gso_par, "orbital_slot_deg", None)
    freq = getattr(gso_par, "center_freq_GHz", None)

    # A compact, plot-friendly label. full_name already encodes link + place
    # in your pipeline, so use it as the primary label and keep coords as a
    # tooltip-style suffix only when they add information.
    label = gso_par.full_name
    coord = None
    if lat is not None and lon is not None:
        coord = f"{lat:.2f}, {lon:.2f}"

    return {
        "label": label,
        "coord": coord,
        "orbital_slot_deg": slot,
        "center_freq_GHz": freq,
    }


def load_link_df(results_dir: Path, full_name: str) -> pd.DataFrame | None:
    """Load one per-link CSV (only cn/cni)."""
    path = results_dir / f"gso_per_iteration{STR_SEPARATOR}{full_name}.csv"
    if not path.exists():
        print(f"[WARN] Missing results file: {path}")
        return None
    return pd.read_csv(path, usecols=lambda c: c in ("cn", "cni"))


# ---------------------------------------------------------------------------
# Build (trial, station) -> arrays of η, discovering links from the yaml
# ---------------------------------------------------------------------------

def build_throughput_table() -> tuple[dict, dict]:
    """
    Returns (data, meta).

    data[trial_label][full_name] = {
        "eta_cn", "eta_cni", "ser", "seri", "rel"
    }
    meta[full_name] = {"label", "coord", "orbital_slot_deg", "center_freq_GHz"}
    """
    data: dict[str, dict[str, dict]] = {}
    meta: dict[str, dict] = {}

    for trial in SCENARIO3_TRIALS:
        try:
            res_dir = results_name2dir(trial["res_name"])
            par = load_params_from_results_dir(res_dir)
        except ValueError as e:
            print(f"[WARN] Skipping trial '{trial['label']}': {e}")
            continue

        data[trial["label"]] = {}

        for gso_par in par.gso_links:
            full_name = gso_par.full_name
            meta.setdefault(full_name, station_meta(gso_par))

            df = load_link_df(res_dir, full_name)
            if df is None or "cn" not in df.columns or "cni" not in df.columns:
                print(f"[WARN] No cn/cni for {full_name} "
                      f"in trial {trial['label']}, skipping.")
                continue

            eta_cn = spectral_efficiency(df["cn"].to_numpy())
            eta_cni = spectral_efficiency(df["cni"].to_numpy())
            ser = float(np.mean(eta_cn))
            seri = float(np.mean(eta_cni))
            rel = (ser - seri) / ser * 100.0 if ser > 0 else float("nan")

            data[trial["label"]][full_name] = {
                "eta_cn": eta_cn, "eta_cni": eta_cni,
                "ser": ser, "seri": seri, "rel": rel,
            }

    return data, meta


def _disp(meta: dict, full_name: str) -> str:
    """Axis label for a station: its full_name (already link + place)."""
    return meta.get(full_name, {}).get("label", full_name)


# ---------------------------------------------------------------------------
# Plot 1 — grouped bar: mean η per station (SER vs SERI), one fig per trial
# ---------------------------------------------------------------------------

def plot_avg_throughput_by_station(data: dict, meta: dict, plots_dir: Path):
    for trial_label, stations in data.items():
        if not stations:
            continue
        names = list(stations.keys())
        disp = [_disp(meta, n) for n in names]
        ser = [stations[n]["ser"] for n in names]
        seri = [stations[n]["seri"] for n in names]

        x = np.arange(len(names))
        w = 0.38
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 5))
        ax.bar(x - w / 2, ser, w, label="SER — rain only (C/N)", color="#4C72B0")
        ax.bar(x + w / 2, seri, w, label="SERI — with NGSO (C/(N+I))", color="#DD8452")

        ax.set_ylabel("Mean spectral efficiency η  (bps/Hz)")
        ax.set_title(f"Scenario 3 — mean throughput per station  ·  {trial_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(disp, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8)
        fig.tight_layout()

        safe = trial_label.lower().replace(".", "").replace(" ", "_")
        out = plots_dir / f"bar_avg_throughput_by_station_{safe}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — relative decrease (%) per station, trials grouped, criterion line
# ---------------------------------------------------------------------------

def plot_relative_decrease_by_station(data: dict, meta: dict, plots_dir: Path):
    trials = list(data.keys())
    stations: list[str] = []
    for t in trials:
        for s in data[t]:
            if s not in stations:
                stations.append(s)
    if not stations:
        return

    x = np.arange(len(stations))
    n_tr = len(trials)
    w = 0.8 / max(n_tr, 1)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(stations)), 5))
    for i, t in enumerate(trials):
        vals = [data[t].get(s, {}).get("rel", np.nan) for s in stations]
        ax.bar(x + (i - (n_tr - 1) / 2) * w, vals, w,
               label=t, color=palette[i % len(palette)])

    ax.axhline(CRITERION_PCT, color="red", linestyle="--", linewidth=1.2,
               label=f"Criterion = {CRITERION_PCT:g}%")
    ax.set_ylabel("Relative throughput decrease  (SER−SERI)/SER  [%]")
    ax.set_title("Scenario 3 — long-term throughput degradation per station")
    ax.set_xticks(x)
    ax.set_xticklabels([_disp(meta, s) for s in stations],
                       rotation=30, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = plots_dir / "bar_relative_decrease_by_station.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — CDF of η(C/(N+I)) per station, one fig per trial
# ---------------------------------------------------------------------------

def plot_throughput_cdf_by_station(data: dict, meta: dict, plots_dir: Path):
    cmap = plt.get_cmap("tab10")
    for trial_label, stations in data.items():
        if not stations:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (name, d) in enumerate(stations.items()):
            xv, p = empirical_cdf(d["eta_cni"])
            ax.plot(xv, p, color=cmap(i % 10), linewidth=1.2,
                    label=_disp(meta, name))

        ax.set_xlabel("Delivered spectral efficiency η(C/(N+I))  (bps/Hz)")
        ax.set_ylabel("P(η ≤ x)")
        ax.set_title(f"Scenario 3 — delivered throughput CDF  ·  {trial_label}")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()

        safe = trial_label.lower().replace(".", "").replace(" ", "_")
        out = plots_dir / f"cdf_throughput_by_station_{safe}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — box of instantaneous η per station (worst-case trial by default)
# ---------------------------------------------------------------------------

def plot_throughput_spread(data: dict, meta: dict, plots_dir: Path,
                           prefer_trial: str = "Worst Case"):
    trial_label = prefer_trial if prefer_trial in data else next(iter(data), None)
    if trial_label is None or not data.get(trial_label):
        return
    stations = data[trial_label]
    names = list(stations.keys())
    samples = [stations[n]["eta_cni"] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names)), 5))
    bp = ax.boxplot(samples, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black"))
    cmap = plt.get_cmap("tab10")
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap(i % 10))
        box.set_alpha(0.6)

    ax.set_ylabel("Delivered spectral efficiency η(C/(N+I))  (bps/Hz)")
    ax.set_title(f"Scenario 3 — throughput spread per station  ·  {trial_label}")
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels([_disp(meta, n) for n in names],
                       rotation=30, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    out = plots_dir / "box_throughput_by_station.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5 — station-averaged mean η, grouped by trial
# ---------------------------------------------------------------------------

def plot_avg_throughput_by_trial(data: dict, plots_dir: Path):
    trials = [t for t in data if data[t]]
    if not trials:
        return
    ser_means = [np.mean([d["ser"] for d in data[t].values()]) for t in trials]
    seri_means = [np.mean([d["seri"] for d in data[t].values()]) for t in trials]

    x = np.arange(len(trials))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(trials)), 5))
    ax.bar(x - w / 2, ser_means, w, label="SER — rain only", color="#4C72B0")
    ax.bar(x + w / 2, seri_means, w, label="SERI — with NGSO", color="#DD8452")

    ax.set_ylabel("Station-averaged mean η  (bps/Hz)")
    ax.set_title("Scenario 3 — throughput by trial (averaged over stations)")
    ax.set_xticks(x)
    ax.set_xticklabels(trials, fontsize=9)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = plots_dir / "bar_avg_throughput_by_trial.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary(data: dict, meta: dict, plots_dir: Path):
    CT, CL, CS, CSI, CR = 16, 34, 14, 14, 16
    header = (
        f"{'Trial':<{CT}}  {'Station':<{CL}}  "
        f"{'SER':>{CS}}  {'SERI':>{CSI}}  {'(SER-SERI)/SER %':>{CR}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for trial_label, stations in data.items():
        for name, d in stations.items():
            disp = _disp(meta, name)
            lines.append(
                f"{trial_label:<{CT}}  {disp:<{CL}}  "
                f"{d['ser']:>{CS}.6f}  {d['seri']:>{CSI}.6f}  {d['rel']:>{CR}.6f}"
            )
        lines.append(sep)

    text = "\n".join(lines)
    print("\n" + text + "\n")
    out = plots_dir / "scenario3_throughput_summary.txt"
    out.write_text(text + "\n")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scenario 3 throughput / spectral-efficiency plots"
    )
    parser.add_argument(
        "-o", "--out-dir", default=None,
        help="Where to write plots (default: <first trial results dir>/"
             + OUT_SUBDIR + ")",
    )
    args = parser.parse_args()

    data, meta = build_throughput_table()
    if not any(data.values()):
        raise ValueError(
            "No data loaded. Check SCENARIO3_TRIALS res_names against "
            "results.json, and that each results dir has its .yaml and CSVs."
        )

    if args.out_dir:
        plots_dir = Path(args.out_dir)
    else:
        first_trial = next(t for t in SCENARIO3_TRIALS
                           if t["label"] in data and data[t["label"]])
        plots_dir = results_name2dir(first_trial["res_name"]) / OUT_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_avg_throughput_by_station(data, meta, plots_dir)
    plot_relative_decrease_by_station(data, meta, plots_dir)
    plot_throughput_cdf_by_station(data, meta, plots_dir)
    plot_throughput_spread(data, meta, plots_dir)
    plot_avg_throughput_by_trial(data, plots_dir)
    save_summary(data, meta, plots_dir)


if __name__ == "__main__":
    main()
