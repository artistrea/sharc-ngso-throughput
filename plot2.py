import os
import numpy as np
from pathlib import Path
from sharc.results import Results
import plotly.graph_objects as go
from sharc.post_processor import PostProcessor
post_processor = PostProcessor()

CAMPAIGN_DIR = Path("sharc/campaigns/system4_vs_fs/")
output_dir = "output"

attributes_to_plot = [
    # ("imt_system_antenna_gain", "cdf"),
    # ("imt_system_path_loss", "cdf"),
    # ("system_imt_antenna_gain", "cdf"),
    # ("imt_dl_inr", "cdf"),
    # ("imt_ul_inr", "cdf"),
    ("imt_dl_inr", "ccdf"),
    # ("imt_dl_pfd_external", "ccdf"),
    # ("imt_dl_pfd_external_aggregated", "ccdf"),
    # ("imt_ul_inr", "ccdf"),
]

samples_for_ccdf = [attr[0] for attr in attributes_to_plot if attr[1] == "ccdf"]
samples_for_cdf = [attr[0] for attr in attributes_to_plot if attr[1] == "cdf"]

campaign_ouput_dir = CAMPAIGN_DIR / output_dir
print("Getting results from", campaign_ouput_dir)
ccdf_results = Results.load_many_from_dir(
    campaign_ouput_dir,
    # filter_fn=lambda x: "mss_d2d_to_eess" in x,
    # filter_fn=re.compile(output_dir_regex).search,
    only_latest=True,
    only_samples=samples_for_ccdf)

cdf_results = Results.load_many_from_dir(
    campaign_ouput_dir,
    # filter_fn=lambda x: "mss_d2d_to_eess" in x,
    # filter_fn=re.compile(output_dir_regex).search,
    only_latest=True,
    only_samples=samples_for_cdf)

plots = post_processor.generate_ccdf_plots_from_results(
    ccdf_results,
    cutoff_percentage=1e-8,
    n_bins=200,
)

post_processor.add_plots(plots)

imt_dl_inr_plot = post_processor.get_plot_by_results_attribute_name(
    "imt_dl_inr", plot_type="ccdf")
if imt_dl_inr_plot is not None:
    imt_dl_inr_plot.add_hline(
        y=0.001,
        line_dash="dash",
        line_color="red",
        annotation_text="Protection Criterion (INR = -6 dB, 1 - p = 99,9%)",
        annotation_position="top left",
        annotation_font_size=14,
    )
    imt_dl_inr_plot.add_vline(
        x=-6,
        line_dash="dash",
        line_color="red",
    )
    imt_dl_inr_plot.update_yaxes(
        title_text="CCDF",
    )
    imt_dl_inr_plot.update_xaxes(
        title_text="INR [dB]",
    )

for attr, plot_type in attributes_to_plot:
    # file = HTMLS_DIR / f"{attr}-{plot_type}.html"
    plot = post_processor.get_plot_by_results_attribute_name(attr, plot_type=plot_type)
    if plot is None:
        print("Skipping", attr, plot_type)
        continue
    # Add plot outline and increase font size
    plot.update_xaxes(
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='inside',
        showline=True,
        gridcolor="#DCDCDC",
        gridwidth=1.5,
        title_font=dict(size=16),
        tickfont=dict(size=16),
    )
    plot.update_yaxes(
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='inside',
        showline=True,
        gridcolor="#DCDCDC",
        gridwidth=1.5
    )
    plot.update_layout(
        xaxis_title_font=dict(size=24),
        yaxis_title_font=dict(size=24),
        template="plotly_white"
    )
    plot.update_layout(
        legend=dict(
            font=dict(size=14),
            x=0.01,
            y=-2.0,
            # xanchor='left',
            orientation='h',
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1
        )
    )
    plot.update_layout(width=1400, height=2000)
    # Set color for each beam elevation
    beam_elev_colors = {
        20: "#1f77b4",
        30: "#ff7f0e",
        45: "#2ca02c",
        70: "#d62728",
        80: "#9467bd",
    }

    # for trace in plot.data:
    #     if trace.name:
    #         # match = re.search(r".*Elev\. = (\d+).*", trace.name)
    #         if match:
    #             beam_elev = int(match.group(1))
    #             if beam_elev in beam_elev_colors:
    #                 trace.line.color = beam_elev_colors[beam_elev]

    # Set marker for each load factor
    load_factor_markers = {
        "20": "circle",
        "50": "square",
    }

    # for trace in plot.data:
    #     if trace.name:
    #         match = re.search(r".*Load Factor = ([\d.]+)%;", trace.name)
    #         if match:
    #             load_factor = match.group(1)
    #             if load_factor == "50":
    #                 trace.line.width = 4
    #             else:
    #                 trace.line.width = 2

    # plot.write_html(file=file, include_plotlyjs="cdn", auto_open=auto_open)
    plot.show()
