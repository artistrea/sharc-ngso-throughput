import numpy as np


def es_ant_gain_1428(off_axis, ant_gain, d_over_lmbda):
    off_axis = np.atleast_1d(off_axis)
    g = np.zeros_like(off_axis)

    if d_over_lmbda < 20:
        raise ValueError("ma burro")
    elif d_over_lmbda <= 25:
        g_max = 20*np.log10(d_over_lmbda) + 7.7
        g1 = 29 - 25 * np.log10(95/d_over_lmbda)
        phi_m = 20 / d_over_lmbda * np.sqrt(g_max - g1)
        # unsupported:
        g[off_axis >= 180.] = -5
        g[off_axis < 180.] = -5
        g[off_axis < 80.] = -9
        g[off_axis < 33.1] = 29 - 25 * np.log10(off_axis[off_axis < 33.1])
        g[off_axis < 95 / d_over_lmbda] = g1
        assert phi_m < 95 / d_over_lmbda
        g[off_axis < phi_m] = ant_gain - 2.5e-3 * (d_over_lmbda*off_axis[off_axis < phi_m])**2
    elif d_over_lmbda <= 100:
        g_max = 20*np.log10(d_over_lmbda) + 7.7
        g1 = 29 - 25 * np.log10(95/d_over_lmbda)
        phi_m = 20 / d_over_lmbda * np.sqrt(g_max - g1)
        # unsupported:
        g[off_axis >= 180.] = -9
        g[off_axis < 180.] = -9
        g[off_axis < 120.] = -4
        g[off_axis < 80.] = -9
        g[off_axis < 33.1] = 29 - 25 * np.log10(off_axis[off_axis < 33.1])
        g[off_axis < 95 / d_over_lmbda] = g1
        assert phi_m < 95 / d_over_lmbda
        g[off_axis < phi_m] = ant_gain - 2.5e-3 * (d_over_lmbda*off_axis[off_axis < phi_m])**2
    elif d_over_lmbda > 100:
        g_max = 20*np.log10(d_over_lmbda) + 8.4
        g1 = -1 + 15 * np.log10(d_over_lmbda)
        phi_m = 20 / d_over_lmbda * np.sqrt(g_max - g1)
        phi_r = 15.85 * d_over_lmbda ** -0.6
        # unsupported:
        g[off_axis >= 180.] = -12
        g[off_axis < 180.] = -12
        g[off_axis < 120.] = -7
        g[off_axis < 80.] = -12
        g[off_axis < 34.1] = 34 - 30 * np.log10(off_axis[off_axis < 34.1])
        g[off_axis < 10.] = 29 - 25 * np.log10(off_axis[off_axis < 10.])
        g[off_axis < phi_r] = g1
        assert phi_r < 10.
        assert phi_m < phi_r
        g[off_axis < phi_m] = ant_gain - 2.5e-3 * (d_over_lmbda*off_axis[off_axis < phi_m])**2

    return g


def plot_gain():
    import matplotlib.pyplot as plt
    off_axis = np.linspace(0.01, 180, 5000)

    # eta = 0.5
    D = 0.9
    lmbda = (3e8/18e9)
    # max_g = 10*np.log10(eta) + 20*np.log10(np.pi * D / lmbda)
    max_g = 42.7
    print("max_g", max_g)
    cases = [
        # (22, 34.5),   # 20 <= D/lambda <= 25
        # (50, 41.7),   # 25 < D/lambda <= 100
        # (65.4, 13.2/(3e8/18e9)),  # D/lambda > 100
        (D/lmbda, max_g),  # D/lambda > 100
    ]
    # epfd = pfd - g_max + g(theta) - rain_loss

    plt.figure(figsize=(10, 6))

    for d_over_lmbda, ant_gain in cases:
        gain = es_ant_gain_1428(off_axis, ant_gain, d_over_lmbda)
        plt.plot(
            off_axis,
            gain,
            label=fr"$D/\lambda={d_over_lmbda}$"
        )

    plt.xlim(0, 80)
    plt.ylim(-20, 60)

    plt.xlabel("Off-axis angle (degrees)")
    plt.ylabel("Gain (dBi)")
    plt.title("ITU-R S.1428 Earth Station Antenna Pattern")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_gain()
