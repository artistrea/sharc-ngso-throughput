"""
This module is mostly code from itur package
"""
import warnings

from astropy import units as u
import numpy as np

from itur.utils import EPSILON, prepare_quantity
from itur.models.itu838 import rain_specific_attenuation
from itur.models.itu839 import rain_height
from itur.models.itu1511 import topographic_altitude
from itur.models.itu837 import rainfall_rate


# this function is basically itur library function (MIT license)
# it's a better implementation for our purposes, though
def rain_attenuation_inv_ccdf(lat, lon, f, el, hs=None, R001=None,
                     tau=45, Ls=None):
    lon = np.mod(lon, 360)
    f = prepare_quantity(f, u.GHz, 'Frequency')
    el = prepare_quantity(el, u.deg, 'Elevation angle')
    hs = prepare_quantity(
        hs, u.km, 'Heigh above mean sea level of the earth station')
    R001 = prepare_quantity(R001, u.mm / u.hr, 'Point rainfall rate')
    tau = prepare_quantity(tau, u.one, 'Polarization tilt angle')
    Ls = prepare_quantity(Ls, u.km, 'Slant path length')
    # print("lon", lon)
    # print("f", f)
    # print("el", el)
    # print("hs", hs)
    # print("R001", R001)
    # print("tau", tau)
    # print("Ls", Ls)

    Re = 8500   # Efective radius of the Earth (8500 km)

    if hs is None:
        hs = topographic_altitude(lat, lon).to(u.km).value

    # Step 1: Compute the rain height (hr) based on ITU - R P.839
    hr = rain_height(lat, lon).value

    # Step 2: Compute the slant path length
    if Ls is None:
        Ls = np.where(
            el >= 5, (hr - hs) / (np.sin(np.deg2rad(el))),         # Eq. 1
            2 * (hr - hs) / (((np.sin(np.deg2rad(el)))**2 +
                              2 * (hr - hs) / Re)**0.5 + (np.sin(np.deg2rad(el)))))  # Eq. 2

    # Step 3: Calculate the horizontal projection, LG, of the
    # slant-path length
    Lg = np.abs(Ls * np.cos(np.deg2rad(el)))

    # Obtain the raingall rate, exceeded for 0.01% of an average year,
    # if not provided, as described in ITU-R P.837.
    if R001 is None:
        R001 = rainfall_rate(lat, lon, 0.01).to(u.mm / u.hr).value + EPSILON

    # Step 5: Obtain the specific attenuation gammar using the frequency
    # dependent coefficients as given in ITU-R P.838
    # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.838-3-200503-I!!PDF-E.pdf
    gammar = rain_specific_attenuation(
        R001, f, el, tau).to(
        u.dB / u.km).value

    # Step 6: Calculate the horizontal reduction factor, r0.01,
    # for 0.01% of the time:
    r001 = 1. / (1 + 0.78 * np.sqrt(Lg * gammar / f) -
                 0.38 * (1 - np.exp(-2 * Lg)))

    # Step 7: Calculate the vertical adjustment factor, v0.01,
    # for 0.01% of the time:
    eta = np.rad2deg(np.arctan2(hr - hs, Lg * r001))

    Delta_h = np.where(hr - hs <= 0, EPSILON, (hr - hs))
    Lr = np.where(eta > el, Lg * r001 / np.cos(np.deg2rad(el)),
                  Delta_h / np.sin(np.deg2rad(el)))

    xi = np.where(np.abs(lat) < 36, 36 - np.abs(lat), 0)

    v001 = 1. / (1 + np.sqrt(np.sin(np.deg2rad(el))) *
                 (31 * (1 - np.exp(-(el / (1 + xi)))) *
                  np.sqrt(Lr * gammar) / f**2 - 0.45))

    # Step 8: calculate the effective path length:
    Le = Lr * v001   # (km)

    # Step 9: The predicted attenuation exceeded for 0.01% of an average
    # year
    A001 = gammar * Le   # (dB)

    # Step 10: The estimated attenuation to be exceeded for other
    # percentages of an average year
    def inv_ccdf(p):
        if np.logical_or(p < 0.001, p > 5).any():
            warnings.warn(
                RuntimeWarning('The method to compute the rain attenuation in '
                               'recommendation ITU-P 618-12 is only valid for '
                               'unavailability values between 0.001 and 5'))
        # if p >= 1:
        #     beta = np.zeros_like(A001)
        # else:
        #     beta = np.where(np.abs(lat) >= 36,
        #                     np.zeros_like(A001),
        #                     np.where((np.abs(lat) < 36) & (el > 25),
        #                              -0.005 * (np.abs(lat) - 36),
        #                              -0.005 * (np.abs(lat) - 36) + 1.8 -
        #                              4.25 * np.sin(np.deg2rad(el))))
        beta = np.where(
                p >= 1,
                np.zeros_like(A001),
                np.where(
                    np.abs(lat) >= 36,
                    np.zeros_like(A001),
                    np.where(
                        (np.abs(lat) < 36) & (el > 25),
                        -0.005 * (np.abs(lat) - 36),
                        -0.005 * (np.abs(lat) - 36) + 1.8 - 4.25 * np.sin(np.deg2rad(el))
                    )
                )
            )

        f = (0.655 + 0.033 * np.log(p) - 0.045 * np.log(A001) -
                      beta * (1 - p) * np.sin(np.deg2rad(el)))
        A = A001 * (p / 0.01)**(-f)
        # logA = np.log(A001) - f * np.log(p / 0.01)
        # A = np.exp(logA)

        return np.maximum(A, 0) * u.dB

    return inv_ccdf

