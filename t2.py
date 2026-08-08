import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import matplotlib.pyplot as plt

# Implementation of AST Block2 phased array antenna.
# It's a 96x80 (7680) array


# Load the amplitude taper LUT of the phased array.
# First column is an index, second and thrird are the combinations of Az and El [-60, 60] with 2.5 degrees resolution
data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'Taper_LUT.mat'))
LUT = data['LUT']
# # pattern_data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'E_H_Pattern_0deg_az_45deg_el.mat'))
# pattern_data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'E_H_Pattern_0deg_az_0deg_el.mat'))
# E_pattern_data = pattern_data['E_pattern'].flatten()
# H_pattern_data = pattern_data['H_pattern'].flatten()

def get_weights_2(elev, az):

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


# Some important parameters
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

# Element positions
x = dx * np.arange(-(maxNx - 1) / 2, (maxNx - 1) / 2 + 1)
y = dy * np.arange(-(maxNy - 1) / 2, (maxNy - 1) / 2 + 1)
X, Y = np.meshgrid(x, y, indexing='ij')
beam_pos = np.stack((Y.flatten(), X.flatten(), np.zeros((maxNx * maxNy))))
# Toff = np.array([Y.flatten(), X.flatten(), np.zeros(maxNx * maxNy)])
# pos = np.array([Y.flatten(), X.flatten(), np.zeros(maxNx * maxNy)])

# array pointing at
# target_elevation = -29.96
# target_azimuth = 32.91
target_elevation = 0.0
target_azimuth = 0.0
target_elev_rad = np.deg2rad(target_elevation)
target_az_rad = np.deg2rad(target_azimuth)

# NOTE: Gain adjustment to match AST Block2 antenna patterns
# lingain is calculated from the radiatorfn - not sure how this gain is calculated exactly
lingain = 1.6125


def get_u_vec(el, az):
    """returns the unit vector in the el, az direction"""
    # [uy, ux]
    # return [np.sin(el) * np.sin(az), np.sin(el) * np.cos(az)]
    # [z, y, x]
    return [np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)]


beam_w = np.exp(
    -1j * wave_number * beam_pos.T @ get_u_vec(target_elev_rad, target_az_rad)
)

# This is the amplitude taper, we need to multiply by the element phasing
# TODO: Check for this gambiarra - LUT was created using pos(M, N) I beleve. So we reorganize.
# wts_taper = get_weights_2(target_elevation, target_azimuth)
# beam_w_taper = wts_taper * beam_w
beam_w_taper = beam_w

# # Plot tapper LUT
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# X_taper, Y_taper = np.meshgrid(np.arange(maxNx), np.arange(maxNy))
# ax.plot_surface(X_taper, Y_taper, np.abs(wts_taper.reshape(maxNy, maxNx)), cmap='viridis')

# ax.set_xlabel('X (elements)')
# ax.set_ylabel('Y (elements)')
# ax.set_zlabel('Amplitude')
# ax.set_title('Phased Array Amplitude Taper')
# plt.show()
# exit()

resolution = 0.5  # adjust as needed
El_val = np.arange(-90, 90 + resolution, resolution)
Az_val = np.arange(-90, 90 + resolution, resolution)

# Plot vertical plane
# theta_scan = np.linspace(-np.pi / 2, np.pi / 2, 1000)
theta_scan = np.deg2rad(El_val)
AF = np.zeros_like(theta_scan, dtype=complex)
for i, th in enumerate(theta_scan):
    phase = np.exp(
        1j * wave_number * beam_pos.T @ get_u_vec(th, target_az_rad)
    )
    # print("phase" ,phase)
    # print("lingain", lingain)
    # print("beam_w_taper", beam_w_taper)
    # print("beam_w_taper.shape", beam_w_taper.shape)
    # print("phase.shape", phase.shape)
    AF[i] = lingain * np.dot(beam_w_taper, phase)
    # print("AF[i]", AF[i])
    # exit()

# Normalize and convert to dB
AF_mag = np.abs(AF)
AF_mag = AF_mag
AF_dB = 20 * np.log10(AF_mag + 1e-12)

plt.figure(figsize=(10, 6))
plt.plot(np.rad2deg(theta_scan), AF_dB, label='Array Factor')
# plt.plot(np.rad2deg(theta_scan), E_pattern_data, label='Reference E-Pattern')
plt.axvline(x=target_elevation, color='r', linestyle='--', label=f'Target Elevation ({target_elevation}°)')
plt.xlabel('Theta (degrees)')
plt.ylabel('Magnitude (dB)')
plt.title('Array Factor')
plt.legend()
plt.grid(True)
plt.show()
exit()

# Plot horizontal plane
# phi = np.linspace(-np.pi / 2, np.pi / 2, 500)
phi = np.deg2rad(Az_val)
AF = np.zeros_like(phi, dtype=complex)
for i, ph in enumerate(phi):
    phase = np.exp(
        1j * wave_number * beam_pos.T @ get_u_vec(target_elev_rad, ph))
    AF[i] = lingain * np.dot(beam_w_taper.flatten(), phase.flatten())

# Normalize and convert to dB
AF_mag = np.abs(AF)
AF_mag = AF_mag
AF_dB = 20 * np.log10(AF_mag + 1e-12)

plt.figure(figsize=(10, 6))
plt.plot(np.rad2deg(phi), AF_dB)
plt.plot(np.rad2deg(phi), H_pattern_data, label='Reference H-Pattern')
plt.axvline(x=target_azimuth, color='r', linestyle='--', label=f'Target Azimuth ({target_azimuth}°)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Magnitude (dB)')
plt.title('Array Factor')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3D Array Factor
# theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
# phi = np.linspace(-np.pi / 2, np.pi / 2, 100)
# Theta, Phi = np.meshgrid(theta, phi)
# AF_3D = np.zeros_like(Theta, dtype=complex)

# for i in range(len(theta)):
#     for j in range(len(phi)):
#         phase = np.exp(1j * wave_number * beam_pos.T @ get_u_vec(Theta[j, i], Phi[j, i]))
#         AF_3D[j, i] = np.dot(beam_w_taper, phase)

# AF_3D_dB = 20 * np.log10(np.abs(AF_3D) + 1e-12)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(np.rad2deg(Theta), np.rad2deg(Phi), AF_3D_dB, cmap='viridis')
# ax.set_xlabel('Theta (degrees)')
# ax.set_ylabel('Phi (degrees)')
# ax.set_zlabel('Magnitude (dB)')
# ax.set_title('3D Array Factor')
# plt.show()
