"""
How this module should be used:
- Units should be checked on boundaries between classes/functions;
- After checking, only raw numbers/np.ndarray should be used for performance;

What it should not do:
- Units conversion automatically;
- Hurt performance too badly;

Most libraries are discarded since they do too much.
"""

import typing
from enum import Enum, auto


class u(Enum):
    """
    A simple enum for unit constants.
    """
    # every enum implemented here also has to have
    # its own class in `q` inheriting from Quantity
    dBW = auto()
    dBmW = auto()
    MHz = auto()
    dBmW_per_MHz = auto()


T = typing.TypeVar("T")


class q:
    """
    Implementations for quantities parsing and unit checking
    """
    class Quantity(typing.Generic[T]):
        """
        This class wraps any value with a unit enum.
        It provides a way to safely access the value while expecting a
            specific unit.
        """
        def __init__(self, unsafe_value: T, unit: u):
            """
            Sets the wrapped value and its associated unit.
            """
            self.unsafe_value = unsafe_value
            self.unit = unit

        def value(self, unit: u):
            """
            Returns the stored value
            Throws an error if you expect a unit and the value was wrapped in another unit
            """
            if self.unit != unit:
                raise ValueError(
                    f"Tried parsing value with unit '{self.unit}' as '{unit}'"
                )
            return self.unsafe_value

    # Subclasses for each unit
    # NOTE: yeah, you gotta repeat yourself. Deal with it
    # More specifically, LSPs don't play well with dynamically defining
    # these utilities, so we gotta hardcode them or get fancy
    # NOTE: never get fancy
    class dBW(Quantity[T], typing.Generic[T]):
        def __init__(self, value):
            super().__init__(value, u.dBW)

    class dBmW(Quantity[T], typing.Generic[T]):
        def __init__(self, value):
            super().__init__(value, u.dBmW)

    class MHz(Quantity[T], typing.Generic[T]):
        def __init__(self, value):
            super().__init__(value, u.MHz)

    class dBmW_per_MHz(Quantity[T], typing.Generic[T]):
        def __init__(self, value):
            super().__init__(value, u.dBmW_per_MHz)


if __name__ == "__main__":
    import numpy as np

    def create_spectral_mask(
        freq: q.MHz[np.ndarray],
        band: q.MHz[np.ndarray],
        p_tx: q.dBmW_per_MHz[np.ndarray],
    ) -> (q.MHz[np.ndarray], q.dBmW_per_MHz[np.ndarray]):
        """
        Returns (frequency_bins, psd_at_bins) that defines a spectral mask
            (-inf, ...bins, inf)
            with psd_at_bins[0] being at bin (-inf, bins[0])
        """
        band_mhz = band.value(u.MHz)
        freq_mhz = freq.value(u.MHz)
        p_tx_dBmW_per_mhz = p_tx.value(u.dBmW) - 10 * np.log10(band_mhz)

        delta_f_lim = np.array([0., 5., 10.])
        f_lims = np.concatenate((
            (freq_mhz - band_mhz / 2) - delta_f_lim[::-1],
            (freq_mhz + band_mhz / 2) + delta_f_lim,
        ))
        spectrl_mask = np.array([
            1., 2.,
        ])
        spectrl_mask = np.concatenate((
            spectrl_mask[::-1], [p_tx_dBmW_per_mhz], spectrl_mask[::-1]
        ))

        return q.MHz(f_lims), q.dBmW_per_MHz(spectrl_mask)

    band = q.MHz(5)
    freq = q.MHz(2100)
    p_tx = q.dBmW(31)
    f_lims, mask = create_spectral_mask(freq, band, p_tx)

    # print("f_lims.unsafe_value", f_lims.unsafe_value)
    # print("f_lims.unit", f_lims.unit)
    # print("mask.unsafe_value", mask.unsafe_value)
    # print("mask.unit", mask.unit)

    # This errors:
    # # mask_dBm_per_MHz = mask.value(u.dBmW)
