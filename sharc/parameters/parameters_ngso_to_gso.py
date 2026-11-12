from dataclasses import dataclass, field
import typing
from warnings import warn

from sharc.parameters.parameters_base import ParametersBase
from sharc.parameters.parameters_orbit import ParametersOrbit


STR_SEPARATOR = "---"
SELECTION_STRATEGIES = ("RAND", "MAX_ELEV", "WC")
# FULL: every NGSO system is co-channel with every GSO link, no scaling.
# FRACTION: interference is scaled by the fraction of the victim band
# covered by the system's emission band.
BAND_OVERLAP_MODES = ("FULL", "FRACTION")
MIGRATION_SCRIPT = "scenarios/migrate_to_ngso_systems.py"


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

    def get_band_MHz(self) -> tuple[float, float]:
        """(low, high) edges of the victim receive band [MHz]."""
        fc = self.center_freq_GHz * 1e3
        return (fc - self.bandwidth_MHz / 2, fc + self.bandwidth_MHz / 2)

    @property
    def full_name(self) -> str | None:
        if None in [self.label, self.earth_station.label]:
            return None

        return ParametersGSO._get_full_name(
            self.label, self.earth_station.label
        )

    @staticmethod
    def _get_full_name(label, earth_station_label):
        return label + STR_SEPARATOR + earth_station_label

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
    One NGSO system (e.g. one operator): one or more orbital shells plus the
    transmit model and coordination parameters.

    All shells of a system share the same n_co_channel budget. Different
    systems select their co-channel satellites independently, and their
    interference is aggregated at the GSO earth station.
    """
    section_name: str = "ngso_systems"
    nested_parameters_enabled: bool = True

    # Identifies the system in results columns (e.g. "i---<label>").
    label: str = None

    # Number of co-frequency satellites considered in the worst-case
    # interference calculation.
    n_co_channel: int = None
    gso_protection_avoidance_angle: float = None

    # Optional overrides. If None, the top-level ngso2gso values are used.
    selection_strategy: str = None
    minimum_elevation: float = None

    # Emission band. Only used, and then required, when
    # ngso2gso.band_overlap is "FRACTION".
    center_freq_GHz: float = None
    bandwidth_MHz: float = None

    def band_overlap_fraction(self, gso: "ParametersGSO") -> float:
        """
        Fraction of the victim (GSO link) receive bandwidth covered by
        this system's emission band, between 0 and 1.
        """
        if self.center_freq_GHz is None or self.bandwidth_MHz is None:
            raise ValueError(
                f"NGSO system '{self.label}' has no emission band "
                "(center_freq_GHz, bandwidth_MHz)"
            )
        fc = self.center_freq_GHz * 1e3
        lo, hi = fc - self.bandwidth_MHz / 2, fc + self.bandwidth_MHz / 2
        v_lo, v_hi = gso.get_band_MHz()
        overlap_MHz = max(0.0, min(hi, v_hi) - max(lo, v_lo))
        return overlap_MHz / gso.bandwidth_MHz

    orbits: list = field(default_factory=lambda: [ParametersOrbit()])
    tx_model: ParametersNGSOTxModel = field(
        default_factory=ParametersNGSOTxModel
    )

    def validate(self, ctx: str):
        if self.label is None:
            raise ValueError(f"{ctx}.label is not set")
        if STR_SEPARATOR in self.label:
            raise ValueError(
                f"{ctx}.label must not contain '{STR_SEPARATOR}'"
            )
        if self.n_co_channel is None or self.n_co_channel < 1:
            raise ValueError(f"{ctx}.n_co_channel must be >= 1.")
        if self.selection_strategy not in SELECTION_STRATEGIES:
            raise ValueError(
                f"{ctx}.selection_strategy is '{self.selection_strategy}'"
            )
        if self.minimum_elevation is None:
            raise ValueError(f"{ctx}.minimum_elevation must be defined")
        if (self.center_freq_GHz is None) != (self.bandwidth_MHz is None):
            raise ValueError(
                f"{ctx}: set both center_freq_GHz and bandwidth_MHz, or neither"
            )
        if self.center_freq_GHz is not None:
            if self.center_freq_GHz <= 0:
                raise ValueError(f"{ctx}.center_freq_GHz must be positive.")
            if self.bandwidth_MHz <= 0:
                raise ValueError(f"{ctx}.bandwidth_MHz must be positive.")
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
    version = 1

    selection_strategy: (
        typing.Literal["RAND"]
        | typing.Literal["MAX_ELEV"]
        | typing.Literal["WC"]
    ) = None

    # --- Simulation control ---
    seed: int = 1251
    delta_t_s: float = 1.0          # time step [s]
    min_t_s: float = 0.0            # simulation start [s]
    max_t_s: float = 1000.0         # simulation end [s]
    batch_size: int = 100           # flush results every N steps
    ref_bandwidth_Hz: float = 1e6   # reference bandwidth for noise / PFD

    # How NGSO interference is scaled by frequency overlap. See
    # BAND_OVERLAP_MODES. "FRACTION" must be chosen explicitly so that the
    # scaling is never applied without the user asking for it.
    band_overlap: (
        typing.Literal["FULL"] | typing.Literal["FRACTION"]
    ) = "FULL"
    # With "FRACTION", an NGSO system whose band does not overlap a GSO
    # link contributes no interference to it. That is rejected unless
    # this is set, since it is usually a typo in a frequency.
    allow_zero_band_overlap: bool = False

    minimum_elevation: float = None

    # --- Sub-parameters ---
    # One entry per NGSO system (operator).
    ngso_systems: list[ParametersNGSO] = field(
        default_factory=lambda: [ParametersNGSO()]
    )

    # List allows multiple GSO victims in one run.
    # load_subparameters handles list[ParametersBase] recursively already.
    gso_links: list[ParametersGSO] = field(
        default_factory=lambda: [ParametersGSO()]
    )

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

        labels_set = set()
        for i, gso in enumerate(self.gso_links):
            gso.validate(f"{ctx}.gso_links[{i}]")
            if gso.full_name in labels_set:
                raise ValueError(
                    "More than one GSO Earth Station has the same label"
                )
            labels_set.add(gso.full_name)

        if self.version < 2:
            if self.selection_strategy is not None:
                raise ValueError("AOPASA")

            self.selection_strategy = "RAND"

        if self.selection_strategy not in SELECTION_STRATEGIES:
            raise ValueError(
                f"{ctx}.selection_strategy is '{self.selection_strategy}'"
            )

        if not self.ngso_systems:
            raise ValueError(f"{ctx}.ngso_systems must not be empty")
        sys_labels = set()
        for i, sys in enumerate(self.ngso_systems):
            if sys.selection_strategy is None:
                sys.selection_strategy = self.selection_strategy
            if sys.minimum_elevation is None:
                sys.minimum_elevation = self.minimum_elevation
            sys.validate(f"{ctx}.ngso_systems[{i}]")
            if sys.label in sys_labels:
                raise ValueError(
                    f"{ctx}.ngso_systems has duplicated label '{sys.label}'"
                )
            sys_labels.add(sys.label)

        self._validate_band_overlap(ctx)

        super().validate(ctx)

    def _validate_band_overlap(self, ctx: str):
        if self.band_overlap not in BAND_OVERLAP_MODES:
            raise ValueError(
                f"{ctx}.band_overlap is '{self.band_overlap}', "
                f"must be one of {BAND_OVERLAP_MODES}"
            )

        if self.band_overlap == "FULL":
            with_band = [
                sys.label for sys in self.ngso_systems
                if sys.center_freq_GHz is not None
            ]
            if with_band:
                warn(
                    f"NGSO systems {with_band} set center_freq_GHz/"
                    f"bandwidth_MHz, but {ctx}.band_overlap is 'FULL', so "
                    "those values are ignored and no frequency scaling is "
                    f"applied. Set {ctx}.band_overlap: FRACTION to scale "
                    "interference by the fraction of each GSO link's band "
                    "they cover."
                )
            if self.allow_zero_band_overlap:
                raise ValueError(
                    f"{ctx}.allow_zero_band_overlap only applies when "
                    f"{ctx}.band_overlap is 'FRACTION'."
                )
            return

        without_band = [
            sys.label for sys in self.ngso_systems
            if sys.center_freq_GHz is None
        ]
        if without_band:
            raise ValueError(
                f"{ctx}.band_overlap is 'FRACTION', so every NGSO system "
                "must set center_freq_GHz and bandwidth_MHz. Missing in: "
                f"{without_band}"
            )

        no_overlap = [
            f"{sys_label} -> {link_label}"
            for link_label, per_sys in self.band_overlap_fractions().items()
            for sys_label, frac in per_sys.items()
            if frac == 0.0
        ]
        if no_overlap and not self.allow_zero_band_overlap:
            raise ValueError(
                "These NGSO systems do not overlap these GSO links in "
                f"frequency, so they would add no interference: {no_overlap}. "
                "Check the frequencies, or set "
                f"{ctx}.allow_zero_band_overlap: true if this is intended."
            )
        if no_overlap:
            warn(
                "NGSO systems with no frequency overlap (no interference "
                f"added): {no_overlap}"
            )

    def band_overlap_fractions(self) -> dict[str, dict[str, float]]:
        """
        Interference scale factor (linear, 0 to 1) per GSO link full name
        and NGSO system label. All 1.0 when band_overlap is "FULL".
        """
        return {
            gso.full_name: {
                sys.label: (
                    1.0 if self.band_overlap == "FULL"
                    else sys.band_overlap_fraction(gso)
                )
                for sys in self.ngso_systems
            }
            for gso in self.gso_links
        }

    def load_parameters_from_file(self, config_file, quiet=True):
        # This override only exists to give a helpful error for param files
        # still in the old single-system format (`ngso2gso.ngso`), which was
        # replaced by the `ngso2gso.ngso_systems` list.
        # Without it, ParametersBase rejects the unknown `ngso` key with a
        # generic "parameter doesn't exist" error, which doesn't tell the
        # user that a migration script exists. Loading itself is unchanged.
        import yaml
        with open(config_file, "r") as f:
            raw = yaml.safe_load(f)
        sec = (raw or {}).get(self.section_name) or {}
        if "ngso" in sec:
            raise ValueError(
                f"{config_file} uses the old '{self.section_name}.ngso' "
                "format. Migrate it with:\n"
                f"    python {MIGRATION_SCRIPT} {config_file}"
            )
        super().load_parameters_from_file(config_file, quiet)
