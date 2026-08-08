"""
Script for post-processing and plotting IMT HIBS RAS 2600 MHz simulation results.
Adds legends to result folders and generates plots using SHARC's PostProcessor.
"""
import os
from pathlib import Path
from sharc.results import Results
import plotly.graph_objects as go
from sharc.post_processor import PostProcessor
import numpy as np  # <- garantir esse import no topo


## Graphics adjustments
cutoff_percentage = 0.001
shift_scale = -10 * np.log10(6000 / (3 * 57)) + 6   # Segment Factor + Filtro
legenda_INR_potencia = "INR [dB]"

# Change default legent to the shifited
post_processor = PostProcessor()

valid_patterns = []

campaign_base_dir = str(
    (Path(__file__).parent / "sharc/campaigns/FSS_E-s_Novas simulações").resolve()
)
print("campaign_base_dir", campaign_base_dir)

many_results = Results.load_many_from_dir(
    os.path.join(
        campaign_base_dir,
        "output_1.2"),
    only_latest=True,
    only_samples=[
        "imt_system_antenna_gain",
        # "imt_bs_antenna_gain",
        # "system_imt_antenna_gain",
        # "imt_system_path_loss",
        # "system_dl_interf_power_per_mhz"
    ],
)

post_processor.add_results(many_results)

plots = post_processor.generate_ccdf_plots_from_results(
    many_results, cutoff_percentage=cutoff_percentage
)

post_processor.add_plots(plots)

# Plot every plot:
for plot in plots:
    plot.show()
