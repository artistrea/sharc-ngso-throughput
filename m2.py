import numpy as np
import matplotlib.pyplot as plt
from sharc.support.geometry import SimulatorGeometry
from sharc.support.sharc_geom import polar_to_cartesian

if __name__ == "__main__":
    # victim_geoms = SimulatorGeometry(1000)
    victim_geoms = SimulatorGeometry(1)
    target_geom = SimulatorGeometry(1)
    target_geom.set_global_coords(
        np.array([0.]),
        np.array([0.]),
        np.array([0.]),
    )
    victim_geoms.set_global_coords(
        np.array([0.]),
        np.array([0.]),
        np.array([0.]),
    )

    global_lla = (0, 0, 0)
    sat_geom = SimulatorGeometry(
        1, True, global_cs=global_lla
    )

    sat_geom.set_local_coord_sys(
        np.array([0.]),
        np.array([0.]),
        np.array([0.]),
    )
    sat_geom.set_local_coords(
        np.array([0.]),
        np.array([0.]),
        np.array([5e3]),
        np.array([0.]),
        np.array([-90.]), # pointing at nadir
    )

    # beam diameter in km
    CellDiameter = 48
    # center carrier frequency in Hz
    fc = 890e6

    # Constants
    SPEED_OF_LIGHT = 299792458

    # wavelength in m
    wavelength = SPEED_OF_LIGHT / fc
    wave_number = 2 * np.pi / wavelength

    # Array parameters
    maxNx = 96
    maxNy = 80
    dx = 0.161  # wavelength spacing (adjust as needed)
    dy = 0.161  # wavelength spacing (adjust as needed)

    x = dx * np.arange(-(maxNx - 1) / 2, (maxNx - 1) / 2 + 1)
    y = dy * np.arange(-(maxNy - 1) / 2, (maxNy - 1) / 2 + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # we want to make it so that array is in the satellite y-z plane
    beam_pos_local = np.stack((np.zeros((maxNx * maxNy)), X.flatten(), Y.flatten()))
    # mapping satellite local to ENU (90deg rotation around y)
    # x = z
    # y = y
    # z = -x
    beam_pos_enu = np.stack((beam_pos_local[2], beam_pos_local[1], -beam_pos_local[0]), axis=-1)
    print("beam_pos_enu.shape", beam_pos_enu.shape)

    # geometry holds many stations geometries. We need to specify the station
    # we should get
    # TODO: better reference?
    this_geom_idx = 0
    beam_pos = sat_geom._vec_local2global(beam_pos_enu, translate=False, permutate=True)
    beam_pos = beam_pos[this_geom_idx].T # to get (3, N)
    # print("beam_pos.shape", beam_pos.shape)
    # exit()
    target_elev, target_azim = sat_geom.get_global_pointing_vector_to(target_geom)
    target_elev, target_azim = target_elev[this_geom_idx], target_azim[this_geom_idx]
    # print("target_elev, target_azim", target_elev, target_azim)
    # exit()

    phi, theta = sat_geom.get_global_pointing_vector_to(victim_geoms)
    phi, theta = phi[this_geom_idx], theta[this_geom_idx]
    pointn_vec = np.stack(polar_to_cartesian(1, target_azim, target_elev))
    pointn_vec = pointn_vec.flatten()
    assert pointn_vec.shape == (3,)
    # print(pointn_vec)
    print("pointn_vec.shape", pointn_vec.shape)
    print("beam_pos.T.shape", beam_pos.T.shape)
    beam_w = np.exp(
        -1j * wave_number * beam_pos.T @ pointn_vec
    )
    print("beam_w", beam_w)
    print("beam_pos", beam_pos)
    print("beam_phi", target_azim)
    print("beam_theta", target_elev)
    print("pointn_vec", pointn_vec)
    print("wavenumber", wave_number)
    # print("beam_w.shape", beam_w.shape)
    # exit()
    beam_w_taper = beam_w
    lingain = 1
    def calculate_gains(phi, theta):
        num_calcs = len(phi)
        AF = np.zeros(num_calcs, dtype=complex)
        for i in range(num_calcs):
            phase = np.exp(
                1j * wave_number * beam_pos.T @ np.array(polar_to_cartesian(
                    1,
                    phi[i],
                    theta[i],
                ))
            )
            # print("phase", phase)
            AF[i] = lingain * np.dot(beam_w_taper, phase)

        # Normalize and convert to dB
        AF_mag = np.abs(AF)
        AF_mag = AF_mag
        AF_dB = 20 * np.log10(AF_mag + 1e-12)

        return AF_dB

    theta = np.linspace(-90., 90., 360)
    phi = np.zeros_like(theta)
    gains = calculate_gains(phi, theta)

    plt.figure(figsize=(10, 6))
    plt.plot(theta, gains, label='Array Factor')
    # plt.plot(np.rad2deg(theta_scan), E_pattern_data, label='Reference E-Pattern')
    plt.axvline(x=target_elev, color='r', linestyle='--', label=f'Target Elevation ({target_elev}°)')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Array Factor')
    plt.legend()
    plt.grid(True)

    # phi = np.linspace(-180., 180., 360)
    phi = np.linspace(0., 360., 360)
    theta = np.zeros_like(phi)
    gains = calculate_gains(phi, theta)

    plt.figure(figsize=(10, 6))
    plt.plot(phi, gains, label='Array Factor')
    # plt.plot(np.rad2deg(theta_scan), E_pattern_data, label='Reference E-Pattern')
    plt.axvline(x=target_azim, color='r', linestyle='--', label=f'Target Azimuth ({target_azim}°)')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Array Factor')
    plt.legend()
    plt.grid(True)
    plt.show()

