from dataclasses import dataclass
from functools import cached_property
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

    # Spatial discretization step along thickness (x) [m]
    dx: float = 5e-6

    # Це перевірка на те, що все поділиться націло і сітка буде рівномірна
    def __post_init__(self):
        for name in ("L_n", "L_s", "L_p"):
            n = getattr(self, name) / self.dx
            if not np.isclose(n, round(n)):
                raise ValueError(f"{name} must be a multiple of dx")

    # Particle radii [m]
    R_n: float = 10e-6
    R_p: float = 10e-6

    # Porosity (volume fraction of liquid electrolyte)
    eps_n: float = 0.3
    eps_s: float = 0.4
    eps_p: float = 0.3

    # Bruggemann's coefficients
    brug: float = 1.5
    @property
    def B_n(self) -> float: return self.eps_n ** self.brug
    @property
    def B_s(self) -> float: return self.eps_s ** self.brug
    @property
    def B_p(self) -> float: return self.eps_p ** self.brug

    # Number of points in zones
    @property
    def N_x_n(self) -> int: return int(np.round(self.L_n / self.dx))

    @property
    def N_x_s(self) -> int: return int(np.round(self.L_s / self.dx))

    @property
    def N_x_p(self) -> int: return int(np.round(self.L_p / self.dx))

    @property
    def N_x_total(self) -> int: return self.N_x_n + self.N_x_s + self.N_x_p

    # Dynamic grids
    @cached_property
    def x_e(self) -> np.ndarray:
        return (np.arange(self.N_x_total) + 0.5) * self.dx

    @cached_property
    def r_n(self) -> np.ndarray:
        return np.linspace(0, self.R_n, self.N_r_n)

    @cached_property
    def r_p(self) -> np.ndarray:
        return np.linspace(0, self.R_p, self.N_r_p)

@dataclass
class BatteryParameters:
    # Electrochemical parameters based on Marquis et al. (2019)
    F: float = 96485.33
    R: float = 8.31446
    T_ref: float = 298.15

    # Kinetic coefficients converted to BV
    K_n: float = 8.19e-5
    K_p: float = 5.03e-6

    # Reference diffusion coefficients
    D_s_n_ref: float = 3.9e-14
    D_s_p_ref: float = 1.0e-13
    D_e_ref: float = 5.34e-10

    # Conductivities [S/m]
    sigma_n: float = 100.0
    sigma_p: float = 10.0
    sigma_e: float = 1.1

    # Electrochemistry
    t_plus: float = 0.4
    c_n_max: float = 24983.0 # Graphite anode
    c_p_max: float = 51218.0 # NMC cathode [mol/m^3]
    c_e_0: float = 1000.0

    # Activation energy (Arrhenius) [J/mol]
    Ea_Ds_n: float = 35000.0
    Ea_Ds_p: float = 29000.0
    Ea_De: float = 15000.0
    Ea_sei: float = 38000.0

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
    L_sei_init: float = 5e-9  # SEI initial Thickness [m]
    T_ref: float = 298.15       # [K]

    D_sol_ref: float = 1.0e-21      # [m²/s]
    Ea_k_sei: float = 40000.0       # [J/mol]
    Ea_D_sol: float = 67000.0       # [J/mol]

    U_sei: float = 0.4        # [V], provisional
    alpha_sei: float = 0.5    # [-]
    c_sol_e: float = 4500.0   # [mol/m³], provisional
    sigma_sei: float = 5e-6
    # SEI formation stoichiometry
    n_sei: float = 1.0        # [mol e- / mol SEI product]

    # Dynamic calculations of specific geometries
    def get_b_n(self, geo: BatteryGeometry) -> float:
        return 3.0 * (1.0 - geo.eps_n) / geo.R_n

    def get_b_p(self, geo: BatteryGeometry) -> float:
        return 3.0 * (1.0 - geo.eps_p) / geo.R_p

    def get_b_s(self, geo: BatteryGeometry) -> float:
            return 0


@dataclass
class SimulationConfig:

    i_app: float = 2.5       # [A/m²]
    A_cell: float = 0.01     # [m²]
    t_final: float = 3600.0

    theta_n_init: float = 0.8
    theta_p_init: float = 0.6

    c_e_init: float = 1000.0
    T_init: float = 298.15

    @property
    def I_app(self) -> float:
        return self.i_app * self.A_cell

    def initial_solid_concentrations(self, param):
        return (
            self.theta_n_init * param.c_n_max,
            self.theta_p_init * param.c_p_max
        )

    def OCV_n(self, theta):
        raise NotImplementedError("OCV_n is not defined")

    def OCV_p(self, theta):
        raise NotImplementedError("OCV_p is not defined")