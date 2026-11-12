from dataclasses import dataclass
from sharc.parameters.parameters_base import ParametersBase
from warnings import warn


@dataclass
class ParametersOrbit(ParametersBase):
    """
    Parameters for satellite orbit configuration, including planes, satellites per plane, and orbital elements.

    The shape of the constellation (n_planes, sats_per_plane, phasing_deg,
    inclination_deg, perigee_alt_km, apogee_alt_km) has no default and MUST be
    set by the user; `validate` raises a ValueError listing any left unset.
    long_asc_deg, omega_deg and initial_mean_anomaly are optional (default 0).
    """
    _REQUIRED_PARAMETERS = (
        "n_planes",
        "sats_per_plane",
        "phasing_deg",
        "inclination_deg",
        "perigee_alt_km",
        "apogee_alt_km",
    )

    # required
    n_planes: int = None
    sats_per_plane: int = None
    phasing_deg: float = None
    inclination_deg: float = None
    perigee_alt_km: float = None
    apogee_alt_km: float = None

    # optional
    long_asc_deg: float = 0.0
    omega_deg: float = 0.0
    initial_mean_anomaly: float = 0.0

    # by default, three time dependant angles are set to be
    # independent random variables.
    # You can model time as the only random variable instead
    # by setting this parameter to True
    model_time_as_random_variable: bool = False

    # You may specify a time range lower bound [s]
    t_min: float = 0.0

    # You may specify a time range upper bound [s]
    # or let a default be chosen where needed
    t_max: float = None

    def load_parameters_from_file(self, config_file: str):
        """Load parameters from file and validate."""
        super().load_parameters_from_file(config_file)

        self.validate("ParametersOrbit")

    def validate(self, ctx: str):
        """
        Validate the orbit parameters for correctness.

        Parameters
        ----------
        ctx : str
            Context string for error messages.
        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        missing = [
            name for name in self._REQUIRED_PARAMETERS
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"{ctx} is missing required orbit parameter(s): "
                f"{', '.join(missing)}.\n"
                f"Every orbit must explicitly set all of: "
                f"{', '.join(self._REQUIRED_PARAMETERS)}."
            )

        if self.n_planes <= 0 or self.sats_per_plane <= 0:
            raise ValueError(
                f"Invalid {ctx}: n_planes = {self.n_planes} and "
                f"sats_per_plane = {self.sats_per_plane} must both be positive."
            )

        if not (0 <= self.inclination_deg <= 180):
            raise ValueError(
                f"Invalid {ctx}.inclination_deg = {
                    self.inclination_deg}. \
                             Must be in the range [0, 180] degrees.")
        if self.perigee_alt_km < 0:
            raise ValueError(
                f"Invalid {ctx}.perigee_alt_km = {
                    self.perigee_alt_km}. \
                             Altitude must be non-negative.")
        if self.apogee_alt_km < self.perigee_alt_km:
            raise ValueError(
                f"Invalid {ctx}.apogee_alt_km = {
                    self.apogee_alt_km}. \
                             Must be greater than or equal to perigee_alt_km.")
        if not (0 <= self.phasing_deg <= 360):
            raise ValueError(
                f"Invalid {ctx}.phasing_deg = {
                    self.phasing_deg}. \
                             Must be in the range [0, 360] degrees.")

        if self.model_time_as_random_variable:
            if self.t_max is None:
                warn(
                    f"{ctx}.t_max was not set. Default values will be used when needed"
                )
            else:
                if self.t_max < self.t_min:
                    raise ValueError(f"{ctx}.t_max should be >= {ctx}.t_min")
            if self.t_min < 0:
                raise ValueError(f"{ctx}.t_min should be >= 0")

        if not self.model_time_as_random_variable:
            # check that defaults haven't been changed
            if self.t_max != ParametersOrbit.t_max:
                raise ValueError(
                    f"You should only set {ctx}.t_max "
                    f"if {ctx}.model_time_as_random_variable is set to True"
                )
            if self.t_min != ParametersOrbit.t_min:
                raise ValueError(
                    f"You should only set {ctx}.t_min "
                    f"if {ctx}.model_time_as_random_variable is set to True"
                )
