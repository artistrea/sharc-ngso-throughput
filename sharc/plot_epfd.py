import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path

from sharc.parameters.parameters_ngso_to_gso import (
    STR_SEPARATOR, ParametersGSO
)


RESULTS_DIR = "./results-ngso"
RESULTS_JSON_RESUME_PATH = Path(f"{RESULTS_DIR}/results.json")


BAND_LIMITS = {
    "lower-mask": {
        "label":        "Article 22 Limits, Table 22-1B, 1.0m",
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
    "upper-mask": {
        "label":        "Article 22 Limits, Table 22-1C, 0.9m",
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


def empirical_ccdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    P(X >= x) as a percentage.
    Exceedance = (n - rank + 1) / n * 100  so the max maps to 100/n not 0.
    """
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    exceedance_pct = np.arange(n, 0, -1) / n * 100.0
    return sorted_vals, exceedance_pct


def load_link_data(
    results_dir: Path, label: str,
    fields: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    filename = f"gso_per_iteration{STR_SEPARATOR}{label}.csv"
    path = results_dir / filename

    if not path.exists():
        raise ValueError("OPA")
        print(f"[WARN] Missing results file: {path}")

    return pd.read_csv(path, usecols=fields)


def results_name2dir(name: str) -> Path:
    if RESULTS_JSON_RESUME_PATH.exists():
        results_json_resume = json.loads(
            RESULTS_JSON_RESUME_PATH.read_text()
        )
    else:
        raise ValueError("Could not find results resume json")
    if name not in results_json_resume:
        raise ValueError(
            f"Could not find {name} in the results resume json"
        )
    results_dir = results_json_resume[name]['latest_res_dir']

    return (
        Path(results_dir)
        if Path(results_dir).is_absolute()
        else Path(".") / results_dir
    )


def main():
    print(
        "WARN: this script needs to be modified by hand..."
    )

    batches = [
        {
            "to_process": [
                {
                    "t": "link",
                    "res_name": "max_elev_scenario2",
                    "link": "Jupiter 97W CT",
                    "place": "Las Vegas",
                    "add_to_label": ", (Max. Elev. Selection)"
                },
                {
                    "t": "link",
                    "res_name": "rand_scenario2",
                    "link": "Jupiter 97W CT",
                    "place": "Las Vegas",
                    "add_to_label": ", (Rand. Selection)"
                },
                {
                    "t": "link",
                    # "res_name": "wc_scenario2",
                    "res_name": "worst_case_scenario2",
                    "link": "Jupiter 97W CT",
                    "place": "Las Vegas",
                    "add_to_label": ", (Worst Case Selection)"
                },
                {
                    "t": "upper-mask",
                },
                {
                    "t": "lower-mask",
                },
            ],
            "outname": "scenario2-modified-epfd.png"
        },
        # {
        #     "to_process": [
        #         {
        #             "t": "link",
        #             "res_name": "max_elev_scenario1_arc10",
        #             "link": "Jupiter 97W CT",
        #             "place": "Miami",
        #             "add_to_label": ", (Max. Elev. Selection)"
        #         },
        #         {
        #             "t": "link",
        #             "res_name": "rand_scenario1_arc10",
        #             "link": "Jupiter 97W CT",
        #             "place": "Miami",
        #             "add_to_label": ", (Rand. Selection)"
        #         },
        #         {
        #             "t": "link",
        #             "res_name": "wc_scenario1_arc10",
        #             "link": "Jupiter 97W CT",
        #             "place": "Miami",
        #             "add_to_label": ", (Worst Case Selection)"
        #         },
        #         {
        #             "t": "upper-mask",
        #         },
        #         {
        #             "t": "lower-mask",
        #         },
        #     ],
        #     "outname": "scenario1-arc10.png"
        # }
    ]

    for batch in batches:
        to_process = batch["to_process"]
        fig, ax = plt.subplots()
        for p in to_process:
            if p["t"].endswith("mask"):
                band = BAND_LIMITS[p["t"]]
                lim_epfd, lim_exc = limit_curve(p["t"])
                ax.semilogy(
                    lim_epfd, np.maximum(lim_exc, 1e-4),   # clip 0 % for log axis
                    color="black", linewidth=2.0, linestyle="--",
                    label=band["label"],
                    zorder=10,
                )
            else:
                incomplete_label = ParametersGSO._get_full_name(p["link"], p["place"])
                results = load_link_data(
                    results_name2dir(p['res_name']),
                    incomplete_label,
                    fields=["epfd"],
                )
                epfd_ref = results['epfd']
                x, exc_pct = empirical_ccdf(epfd_ref)
                add_to_label = p["add_to_label"]
                label = p["link"] + add_to_label
                ax.semilogy(
                    x, exc_pct,
                    label=label, linewidth=1.2,
                )
        ax.set_xlabel("EPFD  (dBW/m²/MHz)")
        ax.set_ylabel("% time EPFD is exceeded  [P(X ≥ x)]")
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=7, loc="upper right")

        ax.set_xlim(-190., -130.)
        ax.set_ylim(1e-3, 1e2)

        fig.tight_layout()

        outname = batch["outname"]
        out_path = Path(".") / outname
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    # plt.show()


if __name__ == "__main__":
    main()
