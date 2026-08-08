import numpy as np

ue_tx_power = np.array([1., 2., 3.])
ue_interf = np.array([])
bs = 0
coupling_loss_imt = np.array([
                                 [1.,2.,3.],
                                 [1.,2.,3.]
                             ])
# calculate intra system interference
interference_per_beam = ue_tx_power[ue_interf] - \
    coupling_loss_imt[bs, ue_interf]
x = 10 * np.log10(
    np.power(10, 0.1 * interference_per_beam),
)

