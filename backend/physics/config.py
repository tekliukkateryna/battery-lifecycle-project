from dataclasses import dataclass
import numpy as np


@dataclass
class BatteryGeometry:

    # Number of discrete points for spherical particles along the radius (r)
    N_r_n: int = 10
    N_r_p: int = 10

    # Zone thicknesses [m]
    L_n: float = 80e-6
    L_s: float = 25e-6
    L_p: float = 80e-6

    # Particle radii [m]
    R_n: float = 10e-6
    R_p: float = 10e-6

    # Spatial discretization step along thickness (x) [m]
    dx: float = 10e-6

    # Porosity (volume fraction of liquid electrolyte)
    eps_n: float = 0.3
    eps_s: float = 0.4
    eps_p: float = 0.3

    # Bruggemann's coefficients
    B_n: float = 0.3**1.5
    B_s: float = 0.4**1.5
    B_p: float = 0.3**1.5

    # --- Number of points in zones ---
    @property
    def N_x_n(self) -> int:
        return int(np.round(self.L_n / self.dx))

    @property
    def N_x_s(self) -> int:
        return int(np.round(self.L_s / self.dx))

    @property
    def N_x_p(self) -> int:
        return int(np.round(self.L_p / self.dx))

    @property
    def N_x_total(self) -> int:
        return self.N_x_n + self.N_x_s + self.N_x_p

    # --- Dynamic grids ---
    @property
    def x_e(self) -> np.ndarray:
        return (np.arange(self.N_x_total) + 0.5) * self.dx

    @property
    def r_n(self) -> np.ndarray:
        return np.linspace(0, self.R_n, self.N_r_n)

    @property
    def r_p(self) -> np.ndarray:
        return np.linspace(0, self.R_p, self.N_r_p)

@dataclass
class BatteryParameters:
    # Faraday and Gas constant
    F: float = 96485.33
    R: float = 8.31446

    K_n: float = 1.0616e-11
    K_p: float = 6.7164e-12

    # Reference diffusion coefficients at T = 298.15 K [m²/s]
    D_s_n_ref: float = 3.9e-14  # Solid-phase anode
    D_s_p_ref: float = 1.0e-14  # Solid-phase cathode
    D_e_ref: float = 2.6e-10  # Electrolyte

    # Activation energy (Arrhenius) [J/mol]
    Ea_Ds_n: float = 35000.0
    Ea_Ds_p: float = 29000.0
    Ea_De: float = 15000.0
    Ea_sei: float = 38000.0

    # Electrochemistry
    t_plus: float = 0.38  # Li+ ion transfer number
    c_n_max: float = 33133.0  # Graphite anode [mol/m^3]
    c_p_max: float = 51217.0  # NMC cathode [mol/m^3]
    c_e_0: float = 1000.0

    # Thermal Parameters
    mass: float = 0.04
    Cp: float = 1100.0  # Heat Capacity [J/(kg·K)]
    h_cool: float = 12.0  # Convection [W/(m²·K)]
    A_surface: float = 0.05  # Cooling area [m²]
    T_amb: float = 298.15  # Environment [K]

    # SEI Degradation Parameters
    k_sei_ref: float = 1e-12  # Reaction constant [m/s]
    M_sei: float = 0.162  # Molar mass [kg/mol]
    rho_sei: float = 2100.0  # Density [kg/m³]
    delta_sei = 5e-9  # SEI Thickness already [m]

    # Dynamic calculations of specific geometries
    def get_b_n(self, geo: BatteryGeometry) -> float:
        return 3.0 * (1.0 - geo.eps_n) / geo.R_n

    def get_b_p(self, geo: BatteryGeometry) -> float:
        return 3.0 * (1.0 - geo.eps_p) / geo.R_p


class SimulationConfig:

    def __init__(
        self, param: BatteryParameters
    ):
        self.Iapp = 2.5  # Applied current [A]
        self.t_final = 3600.0  # Simulation time [s]

        # Initial concentrations [mol/m³]
        self.c_s_n_init = 25000.0  # Lithium in the anode
        self.c_s_p_init = 10000.0  # Lithium in the cathode
        self.c_e_init = 1000.0  # Electrolyte

        self.T_init = param.T_amb  # Initial temperature [K]