from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Union
import numpy as np


@dataclass
class BatteryGeometry:

    # Number of discrete points for spherical particles along the radius (r)
    N_r_n: int = 10
    N_r_p: int = 10

    # Zone thicknesses [m]
    L_n: float = 100e-6
    L_s: float = 25e-6
    L_p: float = 100e-6

    L_x: float = 0.137
    L_y: float = 0.207

    @cached_property
    def A_cell(self):
        return self.L_x * self.L_y

    @cached_property
    def A_surface(self):
        return 2 * (
            self.A_cell
            + (self.L_n + self.L_s + self.L_p) * (self.L_x + self.L_y)
        )

    @cached_property
    def V_batt(self) -> float:
        return self.A_cell * (
            self.L_n + self.L_s + self.L_p
        )

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
    eps_s: float = 1.0
    eps_p: float = 0.3
    
    eps_active_n: float = 0.6
    eps_active_p: float = 0.5

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
        dr = self.R_n / self.N_r_n
        return (np.arange(self.N_r_n) + 0.5) * dr

    @cached_property
    def r_p(self) -> np.ndarray:
        dr = self.R_p / self.N_r_p
        return (np.arange(self.N_r_p) + 0.5) * dr

@dataclass
class BatteryParameters:
    # Electrochemical parameters based on Marquis et al. (2019)
    F: float = 96485.33
    R: float = 8.31446

    # Kinetic coefficients converted to BV
    K_n: float = 1.63764e-4       # 8.19e-5
    K_p: float = 1.00719e-5      # 5.03e-6

    # Reference diffusion coefficients
    D_s_n_ref: float = 3.9e-14
    D_s_p_ref: float = 1.0e-13
    D_e_ref: float = 2.788e-10

    # Conductivities [S/m]
    sigma_n: float = 100.0
    sigma_p: float = 10.0
    sigma_e: float = 1.1

    # Electrochemistry
    t_plus: float = 0.4
    c_n_max: float = 24983.26 # Graphite anode
    c_p_max: float = 51217.93 # LiCoO2 (LCO) cathode
    c_e_0: float = 1000.0

    # Activation energy (Arrhenius) [J/mol]
    Ea_Ds_n: float = 42770.0
    Ea_Ds_p: float = 18550.0
    Ea_De: float = 37040.0
    Ea_sei: float = 0

    # OCV curves
    # Dualfoil1998    
    def OCV_n(self, theta):
        return (0.194
        + 1.5 * np.exp(-120.0 * theta)
        + 0.0351 * np.tanh((theta - 0.286) / 0.083)
        - 0.0045 * np.tanh((theta - 0.849) / 0.119)
        - 0.035 * np.tanh((theta - 0.9233) / 0.05)
        - 0.0147 * np.tanh((theta - 0.5) / 0.034)
        - 0.102 * np.tanh((theta - 0.194) / 0.142)
        - 0.022 * np.tanh((theta - 0.9) / 0.0164)
        - 0.011 * np.tanh((theta - 0.124) / 0.0226)
        + 0.0155 * np.tanh((theta - 0.105) / 0.029)
        )
    
    def OCV_p(self, theta):
        stretch = 1.062
        theta = stretch * theta

        return (
            2.16216
            + 0.07645 * np.tanh(30.834 - 54.4806 * theta)
            + 2.1581 * np.tanh(52.294 - 50.294 * theta)
            - 0.14169 * np.tanh(11.0923 - 19.8543 * theta)
            + 0.2051 * np.tanh(1.4684 - 5.4888 * theta)
            + 0.2531 * np.tanh((-theta + 0.56478) / 0.1316)
            - 0.02167 * np.tanh((theta - 0.525) / 0.006)
        )

    # Thermal Parameters
    h_cool: float = 10.0  # Convection [W/(m²·K)]
    T_amb: float = 298.15  # Environment [K]
    rho_n: float = 1657.0
    rho_s: float = 397.0
    rho_p: float = 3262.0
    Cp_n: float = 700.0
    Cp_s: float = 700.0
    Cp_p: float = 700.0
    def get_theta_thermal(self, geo: BatteryGeometry) -> float:

        numerator = (
            self.rho_n * self.Cp_n * geo.L_n
            + self.rho_s * self.Cp_s * geo.L_s
            + self.rho_p * self.Cp_p * geo.L_p
        )

        L_total = (
            geo.L_n + geo.L_s
            + geo.L_p
        )

        return numerator / L_total
    dUeq_dT: float = -0.4e-3

    # SEI Degradation Parameters
    k_sei_ref: float = 1e-12  # Reaction constant [m/s]
    M_sei: float = 0.162  # Molar mass [kg/mol]
    rho_sei: float = 2100.0  # Density [kg/m³]
    L_sei_init: float = 5e-09  # SEI initial Thickness [m]
    T_ref: float = 298.15       # [K]

    D_sol_ref: float = 2.5e-22      # [m²/s]
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
        return 3.0 * geo.eps_active_n / geo.R_n

    def get_b_p(self, geo: BatteryGeometry) -> float:
        return 3.0 * geo.eps_active_p / geo.R_p

    def get_b_s(self, geo: BatteryGeometry) -> float:
            return 0

@dataclass
class SimulationConfig:

    I_app: float = 0.680616           # [A]
    t_final: float = 3600.0

    theta_n_init: float = 0.8
    theta_p_init: float = 0.6

    upper_cut_off_V: float = 4.1
    lower_cut_off_V: float = 3.105

    c_e_init: float = 1000.0
    T_init: float = 298.15

    def initial_solid_concentrations(self, param):
        return (
            self.theta_n_init * param.c_n_max,
            self.theta_p_init * param.c_p_max
        )