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
    L_n: float = 8.52e-05
    L_s: float = 1.2e-05
    L_p: float = 7.56e-05

    L_x: float = 0.065
    L_y: float = 1.58

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
    dx: float = 1.2e-6

    # Це перевірка на те, що все поділиться націло і сітка буде рівномірна
    def __post_init__(self):
        for name in ("L_n", "L_s", "L_p"):
            n = getattr(self, name) / self.dx
            if not np.isclose(n, round(n)):
                raise ValueError(f"{name} must be a multiple of dx")

    # Particle radii [m]
    R_n: float = 5.86e-06
    R_p: float = 5.22e-06

    # Porosity
    eps_n: float = 0.25
    eps_s: float = 0.47
    eps_p: float = 0.335
    
    eps_active_n: float = 0.75
    eps_active_p: float = 0.665

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
    K_n: float = 7.036788206342308e-06       # 8.19e-5
    K_p: float = 7.073293977209257e-05      # 5.03e-6

    # Reference diffusion coefficients
    D_s_n_ref: float = 3.3e-14
    D_s_p_ref: float = 4e-15
    D_e_ref: float = 1.7694e-10

    # Conductivities [S/m]
    sigma_n: float = 215.0
    sigma_p: float = 0.18
    sigma_e: float = 0.9487

    # Electrochemistry
    t_plus: float = 0.2594
    c_n_max: float = 33133.0 # Graphite anode
    c_p_max: float = 63104.0 # LiCoO2 (LCO) cathode
    c_e_0: float = 1000.0

    # Activation energy (Arrhenius) [J/mol]
    Ea_Ds_n: float = 0.0
    Ea_Ds_p: float = 0.0
    Ea_De: float = 0.0
    Ea_sei: float = 0

    # OCV curves
    # Chen2020 (LG M50 graphite and NMC)
    def OCV_n(self, theta):
        return (
            1.9793 * np.exp(-39.3631 * theta)
            + 0.2482
            - 0.0909 * np.tanh(29.8538 * (theta - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (theta - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (theta - 0.6103))
        )

    def OCV_p(self, theta):
        return (
            -0.8090 * theta
            + 4.4875
            - 0.0428 * np.tanh(18.5138 * (theta - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (theta - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (theta - 0.3120))
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
    dUeq_dT: float = 0.0

    # SEI Degradation Parameters
    M_sei: float = 0.162                     # [kg/mol] (PyBaMM значення)
    Vm_sei: float = 9.585e-05                # [m³/mol]
    rho_sei: float = M_sei / Vm_sei          # ≈ 1690.14 kg/m³
    L_sei_init: float = 5e-9  # SEI initial Thickness [m]
    T_ref: float = 298.15       # [K]

    D_sol_ref: float = 2.5e-22 
    Ea_k_sei: float = 0.0       # [J/mol]
    Ea_D_sol: float = 0.0       # [J/mol]

    U_sei: float = 0.4        # [V], provisional
    # Bulk solvent concentration [mol.m-3]
    c_sol_e: float = 2636.0   # [mol/m³], provisional
    resistivity: float = 200000
    sigma_sei = 1 / resistivity
    # SEI formation stoichiometry
    n_sei: float = 2.0        # [mol e- / mol SEI product]

    # Dynamic calculations of specific geometries
    def get_b_n(self, geo: BatteryGeometry) -> float:
        return 3.0 * geo.eps_active_n / geo.R_n

    def get_b_p(self, geo: BatteryGeometry) -> float:
        return 3.0 * geo.eps_active_p / geo.R_p

    def get_b_s(self, geo: BatteryGeometry) -> float:
            return 0

@dataclass
class SimulationConfig:

    I_app: float = 5.0           # [A]
    I_charge: float = 1.5
    I_discharge: float = 5.0
    t_final: float = 3600.0

    theta_n_init: float = 0.9013973983641687
    theta_p_init: float = 0.2699987322515213

    upper_cut_off_V: float = 4.2
    lower_cut_off_V: float = 2.5

    c_e_init: float = 1000.0
    T_init: float = 298.15

    def initial_solid_concentrations(self, param):
        return (
            self.theta_n_init * param.c_n_max,
            self.theta_p_init * param.c_p_max
        )