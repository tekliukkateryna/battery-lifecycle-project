from dataclasses import dataclass
from functools import cached_property
import numpy as np


@dataclass
class BatteryGeometry:
    N_r_n: int = 10
    N_r_p: int = 10

    # OKane2022 LG M50 electrode stack [m]
    L_n: float = 85.2e-6
    L_s: float = 12.0e-6
    L_p: float = 75.6e-6

    L_x: float = 0.065    # electrode height
    L_y: float = 1.58     # electrode width

    # For lumped thermal model
    cell_cooling_area: float = 0.00531   # [m2]
    cell_volume: float = 2.42e-5         # [m3]
    L_cc_n: float = 12e-6                # current collector thickness [m]
    L_cc_p: float = 16e-6                # current collector thickness [m]

    dx: float = 1.2e-6

    R_n: float = 5.86e-6
    R_p: float = 5.22e-6

    eps_n: float = 0.25
    eps_s: float = 0.47
    eps_p: float = 0.335
    eps_active_n: float = 0.75
    eps_active_p: float = 0.665

    brug: float = 1.5

    def __post_init__(self):
        for name in ("L_n", "L_s", "L_p"):
            ratio = getattr(self, name) / self.dx
            if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
                raise ValueError(f"{name} must be an integer multiple of dx")

    @cached_property
    def A_cell(self) -> float:
        """Unwound electrode active cross-sectional area [m2]."""
        return self.L_x * self.L_y

    @cached_property
    def A_surface(self) -> float:
        """Exposed cooling area of the cylindrical cell, NOT active area."""
        return self.cell_cooling_area

    @cached_property
    def V_batt(self) -> float:
        """Physical cell volume for the lumped thermal model [m3]."""
        return self.cell_volume

    @property
    def B_n(self) -> float:
        return self.eps_n ** self.brug

    @property
    def B_s(self) -> float:
        return self.eps_s ** self.brug

    @property
    def B_p(self) -> float:
        return self.eps_p ** self.brug

    @property
    def N_x_n(self) -> int:
        return int(round(self.L_n / self.dx))

    @property
    def N_x_s(self) -> int:
        return int(round(self.L_s / self.dx))

    @property
    def N_x_p(self) -> int:
        return int(round(self.L_p / self.dx))

    @property
    def N_x_total(self) -> int:
        return self.N_x_n + self.N_x_s + self.N_x_p

    @cached_property
    def x_e(self) -> np.ndarray:
        return (np.arange(self.N_x_total) + 0.5) * self.dx

    @cached_property
    def r_n(self) -> np.ndarray:
        return (np.arange(self.N_r_n) + 0.5) * self.R_n / self.N_r_n

    @cached_property
    def r_p(self) -> np.ndarray:
        return (np.arange(self.N_r_p) + 0.5) * self.R_p / self.N_r_p


@dataclass
class BatteryParameters:
    F: float = 96485.33
    R: float = 8.31446
    T_ref: float = 298.15

    # Chen2020 mass-action Butler-Volmer exchange-current prefactors
    m_n_ref: float = 6.48e-7
    m_p_ref: float = 3.42e-6
    Ea_k_n: float = 35000.0
    Ea_k_p: float = 17800.0

    def K_n(self, T: float) -> float:
        return (self.m_n_ref * np.sqrt(self.c_e_0) * self.c_n_max / self.F
                * np.exp(self.Ea_k_n / self.R * (1/self.T_ref - 1/T)))

    def K_p(self, T: float) -> float:
        return (self.m_p_ref * np.sqrt(self.c_e_0) * self.c_p_max / self.F
                * np.exp(self.Ea_k_p / self.R * (1/self.T_ref - 1/T)))

    D_s_n_ref: float = 3.3e-14
    D_s_p_ref: float = 4e-15
    Ea_Ds_n: float = 30300.0
    Ea_Ds_p: float = 25000.0

    Ea_De: float = 17000.0
    Ea_sigma_e: float = 17000.0

    sigma_n: float = 215.0
    sigma_p: float = 0.18
    t_plus: float = 0.2594
    c_n_max: float = 33133.0
    c_p_max: float = 63104.0
    c_e_0: float = 1000.0

    # Thermal (lumped) parameters: collectors included in heat capacity
    h_cool: float = 10.0
    T_amb: float = 298.15
    rho_n: float = 1657.0
    rho_s: float = 397.0
    rho_p: float = 3262.0
    Cp_n: float = 700.0
    Cp_s: float = 700.0
    Cp_p: float = 700.0
    rho_cc_n: float = 8960.0
    rho_cc_p: float = 2700.0
    Cp_cc_n: float = 385.0
    Cp_cc_p: float = 897.0
    # OKane2022 specifies zero OCP entropic changes for both electrodes.
    dUeq_dT: float = 0.0

    k_sei_ref: float = 1e-12             # [m/s]
    D_sol_ref: float = 2e-18             # EC diffusivity [m2/s]
    c_sol_e: float = 4541.0              # initial EC concentration [mol/m3]

    Ea_k_sei: float = 38000.0
    Ea_D_sol: float = 38000.0

    U_sei: float = 0.4
    alpha_sei: float = 0.5
    L_sei_init: float = 5e-9
    Vm_sei: float = 9.585e-5           # [m3/mol SEI]
    n_sei: float = 1.0                 
    resistivity: float = 200000.0     # [Ohm*m]

    @property
    def sigma_sei(self) -> float:
        return 1.0 / self.resistivity

    @staticmethod
    def OCV_n(theta):
        """Chen2020 graphite analytic OCP; see note at top about OKane CSV."""
        x = np.asarray(theta)
        out = (
            1.9793 * np.exp(-39.3631 * x)
            + 0.2482
            - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
        )
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def OCV_p(theta):
        x = np.asarray(theta)
        out = (
            -0.8090 * x
            + 4.4875
            - 0.0428 * np.tanh(18.5138 * (x - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (x - 0.3120))
        )
        return float(out) if out.ndim == 0 else out

    def electrolyte_diffusivity(self, c_e, T):
        x = np.asarray(c_e) / 1000.0
        d_ref = 8.794e-11 * x*x - 3.972e-10 * x + 4.862e-10
        return d_ref * np.exp(self.Ea_De / self.R * (1/self.T_ref - 1/T))

    def electrolyte_conductivity(self, c_e, T):
        x = np.asarray(c_e) / 1000.0
        k_ref = 0.1297 * x**3 - 2.51 * x**1.5 + 3.329 * x
        return k_ref * np.exp(self.Ea_sigma_e / self.R * (1/self.T_ref - 1/T))

    def get_theta_thermal(self, geo: BatteryGeometry) -> float:
        
        c_heat = geo.A_cell * (
            self.rho_n * self.Cp_n * geo.L_n
            + self.rho_s * self.Cp_s * geo.L_s
            + self.rho_p * self.Cp_p * geo.L_p
            + self.rho_cc_n * self.Cp_cc_n * geo.L_cc_n
            + self.rho_cc_p * self.Cp_cc_p * geo.L_cc_p
        )
        return c_heat / geo.V_batt

    def get_b_n(self, geo: BatteryGeometry) -> float:
        return 3.0 * geo.eps_active_n / geo.R_n

    def get_b_p(self, geo: BatteryGeometry) -> float:
        return 3.0 * geo.eps_active_p / geo.R_p

    def get_b_s(self, geo: BatteryGeometry) -> float:
        return 0.0


@dataclass
class SimulationConfig:
    I_app: float = 5                   # [A] 1C for nominal 5-Ah LG M50
    t_final: float = 3600.0

    theta_n_init: float = 29866.0 / 33133.0
    theta_p_init: float = 17038.0 / 63104.0

    upper_cut_off_V: float = 4.2
    lower_cut_off_V: float = 2.5
    c_e_init: float = 1000.0
    T_init: float = 298.15

    def initial_solid_concentrations(self, param: BatteryParameters):
        return (self.theta_n_init * param.c_n_max,
                self.theta_p_init * param.c_p_max)
