from dataclasses import dataclass, field
from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_orbit import ParametersOrbit


STR_SEPARATOR = "---"


# ---------------------------------------------------------------------------
# GSO Earth Station
# ---------------------------------------------------------------------------

@dataclass
class ParametersGSOEarthStation(ParametersBase):
    """
    Physical location and antenna characteristics of a GSO earth station.
    Separating this from the satellite/link params makes it easier to reuse
    the same link definition across different ground station locations.
    """
    section_name: str = "earth_station"
    nested_parameters_enabled: bool = True

    label: str = None

    lat_deg: float = None
    lon_deg: float = None
    # Altitude is derived from topographic data at runtime if left as None;
    # set explicitly to override (metres).
    alt_m: float = None

    def validate(self, ctx):
        if self.lat_deg is None:
            raise ValueError(f"{ctx}.lat_deg is not set")
        if self.lon_deg is None:
            raise ValueError(f"{ctx}.lon_deg is not set")
        if self.label is None:
            raise ValueError(f"{ctx}.label is not set")


# ---------------------------------------------------------------------------
# GSO Link / Satellite
# ---------------------------------------------------------------------------

@dataclass
class ParametersGSO(ParametersBase):
    """
    One complete GSO reference link: satellite orbital slot, link budget
    parameters, and the associated earth station.

    Multiple instances of this class can coexist inside ParametersNGSO2GSO
    so that interference into several GSO networks can be evaluated in a
    single simulation run.
    """
    section_name: str = "gso"
    nested_parameters_enabled: bool = True

    # Human-readable label — used in results filenames / plot legends.
    label: str = None
    # DO NOT SET. This gets updated automatically to include both the gso link
    # label and the earth station label
    full_name: str = None

    # --- Satellite ---
    orbital_slot_deg: float = None
    center_freq_GHz: float = None
    downlink_freq_MHz: tuple = None   # (low, high) band edges

    # --- Link budget ---
    bandwidth_MHz: float = None
    eirp_dBW_per_carrier: float = None
    peak_rx_antenna_gain: float = None          # dBi
    rx_antenna_size_m: float = None
    g_over_t_dB_per_K: float = None

    # "GW" (gateway) or "CT" (consumer terminal) — kept for reference /
    # downstream filtering; doesn't change the simulation maths here.
    station_type: str = None

    # --- Earth station ---
    earth_station: ParametersGSOEarthStation = field(
        default_factory=ParametersGSOEarthStation
    )

    added_loss: float = 0.0

    def validate(self, ctx: str):
        if self.label is None:
            raise ValueError(f"{ctx}.label is not set")

        if self.center_freq_GHz <= 0:
            raise ValueError(f"{ctx}.center_freq_GHz must be positive.")
        if self.bandwidth_MHz <= 0:
            raise ValueError(f"{ctx}.bandwidth_MHz must be positive.")
        if self.rx_antenna_size_m <= 0:
            raise ValueError(f"{ctx}.rx_antenna_size_m must be positive.")
        if self.station_type not in ("GW", "CT"):
            raise ValueError(
                f"{ctx}.station_type must be 'GW' or 'CT', got '{self.station_type}'."
            )
        super().validate(ctx)

        self.full_name = self.label + STR_SEPARATOR + self.earth_station.label


# ---------------------------------------------------------------------------
# NGSO TX model
# ---------------------------------------------------------------------------

@dataclass
class ParametersNGSOTxModel(ParametersBase):
    """
    Transmit model for the NGSO constellation.

    Only CONSTANT_PFD_AT_GND is implemented today; the structure makes it
    easy to add per-satellite EIRP models later without touching the parent
    class.
    """
    section_name: str = "tx_model"
    nested_parameters_enabled: bool = True

    # Which model to use.  Must match one of the keys in the dispatch logic.
    model: str = "CONSTANT_PFD_AT_GND"

    # --- CONSTANT_PFD_AT_GND parameters ---
    # Power flux density at the reference bandwidth, dBW/m²
    pfd_at_ref_bandwidth_dBW_m2: float = -113.0

    def validate(self, ctx: str):
        supported = {"CONSTANT_PFD_AT_GND"}
        if self.model not in supported:
            raise ValueError(
                f"{ctx}.model '{self.model}' is not supported. "
                f"Choose from: {supported}"
            )
        super().validate(ctx)


# ---------------------------------------------------------------------------
# NGSO constellation
# ---------------------------------------------------------------------------

@dataclass
class ParametersNGSO(ParametersBase):
    """
    NGSO constellation definition: one or more orbital shells plus the
    transmit model and coordination parameters.
    """
    section_name: str = "ngso"
    nested_parameters_enabled: bool = True

    # Number of co-frequency satellites considered in the worst-case
    # interference calculation.
    n_co_channel: int = None
    gso_protection_avoidance_angle: float = None

    orbits: list = field(default_factory=lambda: [ParametersOrbit()])
    tx_model: ParametersNGSOTxModel = field(
        default_factory=ParametersNGSOTxModel
    )

    def validate(self, ctx: str):
        if self.n_co_channel < 1:
            raise ValueError(f"{ctx}.n_co_channel must be >= 1.")
        if not self.orbits:
            raise ValueError(f"{ctx}.orbits must contain at least one orbit.")
        for i, orbit in enumerate(self.orbits):
            orbit.validate(f"{ctx}.orbits[{i}]")
        super().validate(ctx)


# ---------------------------------------------------------------------------
# Top-level simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class ParametersNGSO2GSO(ParametersBase):
    """
    Top-level parameters for an NGSO-into-GSO interference simulation.

    Holds one NGSO constellation and one *or more* GSO reference links.
    Results are computed independently for each GSO link.
    """
    section_name: str = "ngso2gso"
    nested_parameters_enabled: bool = True
    scenario_name: str = None

    # --- Simulation control ---
    seed: int = 1251
    delta_t_s: float = 1.0          # time step [s]
    min_t_s: float = 0.0            # simulation start [s]
    max_t_s: float = 1000.0         # simulation end [s]
    batch_size: int = 100           # flush results every N steps
    ref_bandwidth_Hz: float = 1e6   # reference bandwidth for noise / PFD

    minimum_elevation: float = None

    # --- Sub-parameters ---
    ngso: ParametersNGSO = field(default_factory=ParametersNGSO)

    # List allows multiple GSO victims in one run.
    # load_subparameters handles list[ParametersBase] recursively already.
    gso_links: list = field(default_factory=lambda: [ParametersGSO()])

    def validate(self, ctx: str):
        if self.delta_t_s <= 0:
            raise ValueError(f"{ctx}.delta_t_s must be positive.")
        if self.max_t_s <= self.min_t_s:
            raise ValueError(f"{ctx}.max_t_s must be greater than min_t_s.")
        if self.batch_size < 1:
            raise ValueError(f"{ctx}.batch_size must be >= 1.")
        if self.ref_bandwidth_Hz <= 0:
            raise ValueError(f"{ctx}.ref_bandwidth_Hz must be positive.")
        if not self.gso_links:
            raise ValueError(
                f"{ctx}.gso_links must contain at least one GSO link."
            )
        if self.scenario_name is None:
            raise ValueError(
                f"{ctx}.scenario_name must be defined"
            )
        if self.minimum_elevation is None:
            raise ValueError(
                f"{ctx}.minimum_elevation must be defined"
            )

        self.ngso.validate(f"{ctx}.ngso")

        labels_set = set()
        for i, gso in enumerate(self.gso_links):
            gso.validate(f"{ctx}.gso_links[{i}]")
            if gso.full_name in labels_set:
                raise ValueError(
                    "More than one GSO Earth Station has the same label"
                )
            labels_set.add(gso.full_name)

        super().validate(ctx)
