import numpy as np
import matplotlib.pyplot as plt
from sharc.support.geometry import SimulatorGeometry
from sharc.support.sharc_geom import polar_to_cartesian
from sharc.antenna.antenna import Antenna
from sharc.parameters.constants import SPEED_OF_LIGHT


class AntennaBeamformingSatellite(Antenna):
    """Implements beamforming without taper.

    Considers the geometry for transforming the antenna
    array position and beam direction to the global coord system

    Notes:
    -----
    Only supports one beam, the first beam.
    It is always used in calculations
    """
    def __init__(
        self,
        sat_geom: SimulatorGeometry,
        associated_station_index: int,
        num_el_x: int,
        num_el_y: int,
        dx: float,
        dy: float,
        frequency_MHz: float
    ):
        """Takes a stations full SimulatorGeometry
        and what stations index they should consider inside that geometry
        """
        super().__init__()
        # TODO: have a SingleStationGeometry instead of this hack
        # self.geom = geom
        # self.associated_station_index = associated_station_index
        self.beam_phi = None
        self.beam_theta = None
        wavelength = SPEED_OF_LIGHT / (frequency_MHz * 1e6)
        self.wavenumber = 2 * np.pi / wavelength

        self.calculated_beam_w = None

        x = dx * np.arange(-(num_el_x - 1) / 2, (num_el_x - 1) / 2 + 1)
        y = dy * np.arange(-(num_el_y - 1) / 2, (num_el_y - 1) / 2 + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        # we want to make it so that array is in the satellite y-z plane
        beam_pos_local = np.stack((np.zeros((num_el_x * num_el_y)), X.flatten(), Y.flatten()))
        # mapping satellite local to ENU (90deg rotation around y)
        # x = z
        # y = y
        # z = -x
        beam_pos_enu = np.stack((beam_pos_local[2], beam_pos_local[1], -beam_pos_local[0]), axis=-1)

        # geometry holds many stations geometries. We need to specify the station
        # we should get
        # TODO: better reference?
        beam_pos_simulator = sat_geom._vec_local2global(beam_pos_enu, translate=False, permutate=True)
        beam_pos_simulator = beam_pos_simulator[associated_station_index].T # to get (3, N)

        self.beam_pos = beam_pos_simulator

    def add_beam(self, simulator_phi, simulator_theta):
        if len(self.beams_list) != 0:
            # NOTE: DANGEROUS BEHAVIOR
            # TODO: make this more obvious to the user OR integrate with other parts
            return

        self.beams_list.append((simulator_phi, simulator_theta))
        self.beam_phi = simulator_phi
        self.beam_theta = simulator_theta

    def get_beam_weight(self):
        assert len(self.beams_list) == 1
        if self.calculated_beam_w is not None:
            return self.calculated_beam_w

        pointn_vec = np.array(polar_to_cartesian(1, self.beam_phi, self.beam_theta))
        beam_w = np.exp(
            -1j * self.wavenumber * self.beam_pos.T @ pointn_vec
        )

        beam_w_taper = beam_w

        self.calculated_beam_w = beam_w_taper

        return self.calculated_beam_w

    def calculate_gain(self, **kwargs):
        """
        phi_vec (np.array): azimuth angles [degrees]
        theta_vec (np.array): elevation angles [degrees]
        """
        phi = kwargs["phi_vec"]
        theta = kwargs["theta_vec"]

        num_calcs = len(phi)
        AF = np.zeros(num_calcs, dtype=complex)
        lingain = 1
        beam_w_taper = self.get_beam_weight()

        for i in range(num_calcs):
            phase = np.exp(
                1j * self.wavenumber * self.beam_pos.T @ polar_to_cartesian(
                    1,
                    phi[i],
                    theta[i],
                )
            )
            AF[i] = lingain * np.dot(beam_w_taper, phase)

        # Normalize and convert to dB
        AF_mag = np.abs(AF)
        AF_mag = AF_mag
        AF_dB = 20 * np.log10(AF_mag + 1e-12)

        return AF_dB


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

    ##########################################################
    # what the satellite needs to do:
    global_lla = (0, 0, 0)
    sat_geom = SimulatorGeometry(
        1, True, global_cs=global_lla
    )

    # in GLOBAL coordinates
    # e.g. if we have satellite right above the global reference, then that
    # antenna will be pointing downwards
    # and beamforming pointing forwards is globally pointing dowards
    target_azim, target_elev = 180., 0.

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
        np.array([90.]), # pointing at nadir
    )

    fc = 890 # MHz
    maxNx = 96
    maxNy = 80
    dx = 0.161  # wavelength spacing (adjust as needed)
    dy = 0.161  # wavelength spacing (adjust as needed)
    antenna = AntennaBeamformingSatellite(
        sat_geom, 0,
        maxNx, maxNy, dx, dy,
        fc
    )
    antenna.add_beam(
        target_azim,
        target_elev,
    )
    ##########################################################

    theta = np.linspace(-90., 90., 360)
    phi = np.zeros_like(theta)
    gains = antenna.calculate_gain(
        phi_vec=phi,
        theta_vec=theta
    )

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
    gains = antenna.calculate_gain(
        phi_vec=phi,
        theta_vec=theta
    )

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
