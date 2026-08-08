import numpy as np
import matplotlib.pyplot as plt
from sharc.support.geometry import SimulatorGeometry
from sharc.support.sharc_geom import polar_to_cartesian, cartesian_to_polar
from sharc.antenna.antenna import Antenna
from sharc.parameters.constants import SPEED_OF_LIGHT
import scipy.io
import os

data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'Taper_LUT.mat'))
LUT = data['LUT']
# # pattern_data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'E_H_Pattern_0deg_az_45deg_el.mat'))
pattern_data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'E_H_Pattern_0deg_az_0deg_el.mat'))
E_pattern_data = pattern_data['E_pattern'].flatten()
H_pattern_data = pattern_data['H_pattern'].flatten()

def get_weights_2(az, elev):
    # print("az", az)
    # print("elev", elev)
    radiatingAngle = np.array([az, elev]).reshape(2, 1)
    beamAzEl_LUT = radiatingAngle
    beamAzEl = radiatingAngle
    wts = np.zeros((7680, 1))
    taperVal = np.zeros((7680, 1))
    ival = np.abs(LUT[0, 2])  # lower value for elevation in the table
    lut_res = np.abs(LUT[0, 2]) - np.abs(LUT[1, 2])  # Beamformer LUT resolution (2.5 deg.)

    # Assuming radiatingAngle, beamAzEl_LUT, beamAzEl, range, th_range are inputs
    # and taperVal, wts, wts_taper are outputs to be initialized

    # beamAzEl is the Az and El towards the Earth (can be a victim)
    # beamAzEl_LUT is beamAzEl rotated by roll, yaw and pitch - those are rotations to the antenna - based on local
    # antenna coordinate system

    for i in range(radiatingAngle.shape[0]):
        # TODO: In the original code th_range is a threshold for slant path length. We need to convert to elevation
        # look angle
        # if range[i] > th_range:
        if i > 60.0 or i < -60.0:
            # Just find two LUT bounds arround the actual az and el
            a1 = np.round(beamAzEl_LUT[0, i] / lut_res) * lut_res
            a2 = a1 + lut_res if a1 < beamAzEl_LUT[0, i] else a1 - lut_res
            e1 = np.round(beamAzEl_LUT[1, i] / lut_res) * lut_res
            e2 = e1 + lut_res if e1 < beamAzEl_LUT[1, i] else e1 - lut_res

            az1 = min(a1, a2)
            az2 = max(a1, a2)
            el1 = min(e1, e2)
            el2 = max(e1, e2)

            bid1 = int(np.round((az1 + ival) / lut_res) * (1 + (2 * ival) / lut_res) + np.round((el1 + ival) / lut_res))
            bid2 = bid1 + int(1 + (2 * ival / lut_res))
            bid3 = bid1 + 1
            bid4 = bid2 + 1

            x = np.abs(az1 - beamAzEl_LUT[0, i]) / lut_res
            y = np.abs(el1 - beamAzEl_LUT[1, i]) / lut_res
            inp_taper = (LUT[bid1, 3:] * (1 - x) * (1 - y) +
                         LUT[bid2, 3:] * x * (1 - y) +
                         LUT[bid3, 3:] * (1 - x) * y +
                         LUT[bid4, 3:] * x * y)
            taperVal[:, i] = inp_taper
        else:
            bid = int(np.round((beamAzEl_LUT[0, i] + ival) / lut_res) * (1 + (2 * ival) / lut_res) +
                      np.round((beamAzEl_LUT[1, i] + ival) / lut_res))
            taperVal[:, i] = LUT[bid, 3:]
        # # Compute weights
        # delay = (1 / SPEED_OF_LIGHT) * pos.T @ np.array([
        #     np.sin(np.deg2rad(beamAzEl[1, i])),
        #     np.cos(np.deg2rad(beamAzEl[1, i])) * np.sin(np.deg2rad(beamAzEl[0, i])),
        #     np.cos(np.deg2rad(beamAzEl[1, i])) * np.cos(np.deg2rad(beamAzEl[0, i]))
        # ])
        # new_wts = np.exp(1j * 2 * np.pi * fc * delay)
        # wts[:, i] = new_wts
        # wts_taper = new_wts * taperVal[:, i]

        return taperVal.flatten()

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
        frequency_MHz: float,
        taper_fn=None,
    ):
        """Takes a stations full SimulatorGeometry
        and what stations index they should consider inside that geometry
        """
        super().__init__()
        self.taper_fn = taper_fn
        self.sat_geom = sat_geom
        self.associated_station_index = associated_station_index
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
        X, Y = np.meshgrid(x, y)
        # we want to make it so that array is in the satellite y-z plane
        beam_pos_local = np.stack((np.zeros((num_el_x * num_el_y)), X.flatten(), Y.flatten()))
        # TODO: use rigid body frame for correct fixed axis pointing
        # NOTE: using the current kind of mapping means that the antenna rotates
        # as the satellite moves. This is WRONG
        # local (down, west, north) to ENU
        # x = -y; y = z; z = -x
        beam_pos_enu = np.stack((-beam_pos_local[1], beam_pos_local[2], -beam_pos_local[0]))
        print("beam_pos_local", beam_pos_local)
        print("beam_pos_enu", beam_pos_enu)

        # geometry holds many stations geometries. We need to specify the station
        # we should get with `associated_station_index`
        # TODO: better reference?
        beam_pos_simulator = sat_geom._vec_local2global(beam_pos_enu.T, translate=False, permutate=True)
        beam_pos_simulator = beam_pos_simulator[associated_station_index].T # to get (3, N)

        self.beam_pos = beam_pos_simulator
        print("self.beam_pos", self.beam_pos)
        # exit()

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

        # print("self.beam_phi", self.beam_phi)
        # print("self.beam_theta", self.beam_theta)
        pointn_vec = np.array(polar_to_cartesian(1, self.beam_phi, self.beam_theta))
        # print("pointn_vec", pointn_vec)
        assert pointn_vec.shape == (3,)
        beam_w = np.exp(
            -1j * self.wavenumber * self.beam_pos.T @ pointn_vec
        )

        if self.taper_fn is not None:
            ##### Simulator global -> ENU
            assert pointn_vec.shape == (3,)

            global_pointn = pointn_vec.reshape(1, 3)
            enu = self.sat_geom._vec_global2local(global_pointn, translate=False, permutate=True)
            enu = enu[self.associated_station_index].T
            # print("enu", enu)
            # TODO: use rigid body frame for correct fixed axis pointing
            # NOTE: using ENU as reference means the antenna rotates along orbit
            # ENU to local (down, west, north)
            # x = -z; y = -x; z = y
            local = (-enu[2], -enu[0], enu[1])
            # print("local", local)
            # print("enu", enu)
            # print("local", local)

            _, phi_local, theta_local = cartesian_to_polar(
                local[0],
                local[1],
                local[2],
            )
            # however, the azimuth we really want to consider is angle
            # from the line perpendicular to antenna plane
            # which is y axis, pointing down and 90deg away from
            antenna_off_azim = phi_local
            antenna_off_elev = theta_local

            # pases azim and elev in local coords
            taper = self.taper_fn(antenna_off_azim, antenna_off_elev)
            # exit()
        else:
            taper = 1

        # beam_w_taper = taper * beam_w
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
        # lingain = 2.0
        lingain = 1.6125
        beam_w_taper = self.get_beam_weight()
        # print("beam_w_taper", beam_w_taper)
        # print("wave_number", self.wavenumber)

        for i in range(num_calcs):
            direction = polar_to_cartesian(
                1,
                phi[i],
                theta[i],
            )

            phase = np.exp(
                1j * self.wavenumber * self.beam_pos.T @ direction
            )
            # print("beam_w_taper", beam_w_taper)
            # print("beam_pos", self.beam_pos)
            # print("direction", direction)
            # print("beam_pos.T @ direction", self.beam_pos.T @ direction)
            # print("phase", phase)
            AF[i] = lingain * np.dot(beam_w_taper, phase)
            # print("AF[i]", AF[i])
            # exit()

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

    # aligning ECEF and ENU
    global_lla = (-90., 90., 0.)
    ##########################################################
    # what the satellite needs to do:
    sat_geom = SimulatorGeometry(
        1, True, global_cs=global_lla
    )

    # in GLOBAL coordinates
    # e.g. if we have satellite right above the global reference, then that
    # antenna will be pointing downwards
    # and beamforming pointing forwards is globally pointing dowards
    # WE CAN ALSO ALIGN LOCAL TO GLOBAL

    # sat_geom.set_local_coord_sys(
    #     np.array([0.]),
    #     np.array([0.]),
    #     np.array([0.]),
    # )
    # sat_geom.set_local_coord_sys(
    #     np.array([-90.]),
    #     np.array([90.]),
    #     np.array([0.]),
    # )
    sat_geom.set_local_coord_sys(
        np.array([0.]),
        np.array([180.]),
        np.array([0.]),
    )
    local = np.array([1, 2, 3])
    enu = np.stack((-local[1], local[2], -local[0]))
    glob = sat_geom._vec_local2global(np.array([enu]), translate=False)
    # print("local", local)
    # print("enu", enu)
    # print("glob", glob)
    # exit()
    sat_geom.set_local_coords(
        np.array([0.]),
        np.array([0.]),
        np.array([5e3]),
        np.array([0.]),
        np.array([-90.]), # pointing at nadir
    )

    # target_azim, target_elev = 180., 0.
    # target_azim, target_elev = 0., 0.
    target_azim, target_elev = sat_geom.pointn_azim_global[0], sat_geom.pointn_elev_global[0]

    # sat_geom.set_global_coords(
    #     azim=np.array([0.]),
    #     elev=np.array([0.]),
    # )

    # print("sat_geom.pointn_azim_global", sat_geom.pointn_azim_global)
    # print("sat_geom.pointn_elev_global", sat_geom.pointn_elev_global)

    fc = 890 # MHz
    maxNx = 96
    maxNy = 80
    dx = 0.161  # wavelength spacing (adjust as needed)
    dy = 0.161  # wavelength spacing (adjust as needed)
    antenna = AntennaBeamformingSatellite(
        sat_geom, 0,
        maxNx, maxNy, dx, dy,
        fc,
        taper_fn=get_weights_2
    )
    antenna.add_beam(
        target_azim,
        target_elev,
    )
    ##########################################################

    resolution = 0.5
    theta = np.arange(-90., 90. + resolution, resolution)
    phi = np.zeros_like(theta)
    gains = antenna.calculate_gain(
        phi_vec=phi,
        theta_vec=theta
    )
    # theta_lut = np.linspace(-90., 90., 361)

    plt.figure(figsize=(10, 6))
    plt.plot(theta, gains, label='Array Factor')
    plt.plot((theta), E_pattern_data, label='Reference E-Pattern')
    # plt.plot(np.rad2deg(theta_scan), E_pattern_data, label='Reference E-Pattern')
    plt.axvline(x=target_elev, color='r', linestyle='--', label=f'Target Elevation ({target_elev}°)')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Array Factor')
    plt.legend()
    plt.grid(True)
    plt.show()

    # phi = np.linspace(-180., 180., 361)

    # # phi = np.linspace(0., 360., 361)
    # theta = np.zeros_like(phi)
    # gains = antenna.calculate_gain(
    #     phi_vec=phi,
    #     theta_vec=theta
    # )

    # plt.figure(figsize=(10, 6))
    # plt.plot(phi, gains, label='Array Factor')
    # # plt.plot(np.rad2deg(theta_scan), E_pattern_data, label='Reference E-Pattern')
    # plt.plot((phi), H_pattern_data, label='Reference H-Pattern')
    # plt.axvline(x=target_azim, color='r', linestyle='--', label=f'Target Azimuth ({target_azim}°)')
    # plt.xlabel('Phi (degrees)')
    # plt.ylabel('Magnitude (dB)')
    # plt.title('Array Factor')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
