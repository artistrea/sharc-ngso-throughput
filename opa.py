import matplotlib.pyplot as plt
import numpy as np

random_number_gen = np.random.RandomState(seed=1)
r_min = 35
r_max = 800

SAMPLES = int(1e4)

distance = np.sqrt(
    random_number_gen.random_sample(
        SAMPLES
    ) * (r_max**2 - r_min**2) + r_min**2
)

def get_angle(
    h, x
):
    return 180 - np.rad2deg(np.arctan2(x, h))

h_ue = 1.5
h_bs = 20
angles = get_angle(
    h_bs - h_ue, distance
)

# plt.hist(distance, bins=50)
# plt.show()

plt.hist(angles, bins=50)
plt.show()
