import os
import numpy as np
from pathlib import Path
from sharc.results import Results
import plotly.graph_objects as go
from sharc.post_processor import PostProcessor
post_processor = PostProcessor()

# \ -\ bis/output/output_dc_mss_to_fs_System_525km_altant_20m_azi_90deg_lf_0.1_2025-08-30_01/
BASE1 = Path("sharc/campaigns/system4_vs_fs/")
BASE2 = Path("sharc/campaigns/01_DC_MSS_to_FS - America do Sul - In_Adj - 1944 - bis/")
res1 = Results.load_many_from_dir(
    os.path.join(
        BASE1,
        "output"),
    only_latest=True,
    only_samples=["system_inr", "imt_dl_inr"],
    # filter_fn=filter_fn
)
# res2 = Results.load_many_from_dir(
#     os.path.join(
#         BASE2,
#         "output"),
#     only_latest=True,
#     # filter_fn=filter_fn
# )
# res3 = Results.load_many_from_dir(
#     os.path.join(
#         BASE1,
#         "output2"),
#     only_latest=True,
#     # filter_fn=filter_fn
# )
# res4 = Results.load_many_from_dir(
#     os.path.join(
#         BASE2,
#         "output2"),
#     only_latest=True,
#     # filter_fn=filter_fn
# )
# res5 = Results.load_many_from_dir(
#     os.path.join(
#         BASE2,
#         "output-novo"),
#     only_latest=True,
#     # filter_fn=filter_fn
# )
# res6 = Results.load_many_from_dir(
#     os.path.join(
#         BASE1,
#         "output-novo"),
#     only_latest=True,
#     # filter_fn=filter_fn
# )
# ^: typing.List[Results]
many_results = [
    *res1,
    # *res2,
    # *res3, *res4,
    # *res5, *res6,
]
post_processor.add_results(many_results)
post_processor.add_plot_legend_pattern(
    dir_name_contains="1944 - bis/output",
    legend="v2"
)
post_processor.add_plot_legend_pattern(
    dir_name_contains="1944/output",
    legend="v1"
)
post_processor.add_plot_legend_pattern(
    dir_name_contains="1944 - bis/output-novo",
    legend="v2 - random grid, new lat, lon"
)
post_processor.add_plot_legend_pattern(
    dir_name_contains="1944/output-novo",
    legend="v1 - random grid, new lat, lon"
)
post_processor.add_plot_legend_pattern(
    dir_name_contains="1944 - bis/output2",
    legend="v2 - new lat, lon"
)
post_processor.add_plot_legend_pattern(
    dir_name_contains="1944/output2",
    legend="v1 - new lat, lon"
)
plots = post_processor.generate_ccdf_plots_from_results(
    many_results, cutoff_percentage=0.00001
)
post_processor.add_plots(plots)

# post_processor.get_plot_by_results_attribute_name("system_dl_interf_power").write_html("demo.html")
# post_processor.get_plot_by_results_attribute_name("system_dl_interf_power").show()
# for plot in plots:
#     plot.show()

fig1 = post_processor.get_plot_by_results_attribute_name("system_inr", plot_type="ccdf")
fig2 = post_processor.get_plot_by_results_attribute_name("imt_dl_inr", plot_type="ccdf")
# fig = go.Figure(data=fig1.data + fig2.data)
fig = fig1
for d in fig2.data:
    fig.add_trace(d)
# fig.update_layout(
#     title=f'CCDF Plot for INR',
#     xaxis_title="INR [dB]",
#     yaxis_title="CCDF",
#     yaxis=dict(
#         tickmode="array",
#         tickvals=[1., 0.1, 0.01, 0.001, 0.0001],
#         type="log",
#         range=[
#             np.log10(0.00001),
#             0]),
#     xaxis=dict(
#         tickmode="linear",
#         dtick=5),
#     legend_title="Labels",
#     meta={
#         # "related_results_attribute": attr_name,
#         "plot_type": "ccdf"},
# )
fig.show()

# Atualmente, temos um aparente bug:
# ao utilizar mss dc como imt vs outro sistema, mudar 800 metros de raio da célula
# faz com que o gráfico de cdf de interferência fique extremamente diferente
# para verificar, primeira coisa que fiz foi rodar os arquivos passados com 200 drops,
# usando FSPL. Depois plotar as duas versões

# ao plotar os resultados, é claro que a interferência que chega na estação terrestre
# usando 40.8km de raio aparenta ter uma discretização em 90%
# vou olhar um pouco os dados dos dois e comparar. Coloquei a mesma seed nos dois, então cada drop
# deveria ter interferência mais ou menos parecida... (ou não?)
