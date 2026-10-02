from dataclasses import dataclass
import numpy as np


class OCVProvider:
    """
    Equilibrium Open Circuit Voltage (OCV) curves.
    Standard empirical fits for Graphite anode and NMC cathode.
    """

    @staticmethod
    def ocv_anode(theta: float) -> float:
        # Avoid log/power singularity at boundaries (0 < theta < 1)
        theta = np.clip(theta, 1e-4, 0.9999)
        
        # Empirical fit for graphite (e.g., Prada / Marquis et al.)
        u_n = (
            0.7222 
            + 0.1387 * theta 
            - 0.029 * (theta ** 0.5)
            - 0.0172 / theta 
            + 0.0019 / (theta ** 1.5)
            + 0.2808 * np.exp(0.9 - 15.0 * theta)
            - 0.7984 * np.exp(0.4465 * theta - 0.4108)
        )
        return float(u_n)

    @staticmethod
    def ocv_cathode(theta: float) -> float:
        theta = np.clip(theta, 1e-4, 0.9999)
        
        # Standard NMC stoichiometry fit
        u_p = (
            -0.8090 * theta 
            + 4.4875
            - 0.0428 * np.tanh((theta - 0.6123) / 0.0545)
            - 0.0147 * np.tanh((theta - 1.0000) / 0.0380)
            - 0.3928 * np.tanh((theta - 0.4111) / 0.1901)
            - 0.0131 * np.tanh((theta - 0.0344) / 0.0093)
        )
        return float(u_p)


class ExtendedTerminalVoltage:
    """
    Computes cell terminal voltage considering:
    V = U_p - U_n - eta_p + eta_n + Delta_Phi_electrolyte
    """

    def __init__(self, geo, param, sim):
        self.geo = geo
        self.param = param
        self.sim = sim
        self.ocv = OCVProvider()

    def get_ocv_diff(self, c_s_n_surf: float, c_s_p_surf: float) -> float:
        theta_n = c_s_n_surf / self.param.c_n_max
        theta_p = c_s_p_surf / self.param.c_p_max
        return self.ocv.ocv_cathode(theta_p) - self.ocv.ocv_anode(theta_n)

    def compute_j0(self, c_e_slice: np.ndarray, c_s_surf: float, k_rate: float, c_max: float) -> np.ndarray:
        # Stoichiometry at surface
        theta = c_s_surf / c_max
        theta = np.clip(theta, 1e-4, 0.9999)
        
        # Local exchange current density
        c_e_ratio = np.maximum(c_e_slice / self.param.c_e_0, 0.0)
        return self.param.F * k_rate * np.sqrt(c_e_ratio * theta * (1.0 - theta))

    def compute_reaction_overpotentials(self, c_e: np.ndarray, c_sn_surf: float, c_sp_surf: float, T: float):
        # Surface area specific current densities [A/m^2]
        # Current divided by cross section and active area
        a_s_n = 3.0 * (1.0 - self.geo.eps_n) / self.geo.R_n
        a_s_p = 3.0 * (1.0 - self.geo.eps_p) / self.geo.R_p

        # Apparent surface flux
        j_surf_n = self.sim.Iapp / (self.geo.L_n * a_s_n)
        j_surf_p = -self.sim.Iapp / (self.geo.L_p * a_s_p)

        c_e_n = c_e[: self.geo.N_x_n]
        c_e_p = c_e[self.geo.N_x_n + self.geo.N_x_s :]

        j0_n = np.mean(self.compute_j0(c_e_n, c_sn_surf, self.param.K_n, self.param.c_n_max))
        j0_p = np.mean(self.compute_j0(c_e_p, c_sp_surf, self.param.K_p, self.param.c_p_max))

        # Butler-Volmer inverse hyperbolic sine with factor 2
        thermal_v = (2.0 * self.param.R * T) / self.param.F
        eta_n = thermal_v * np.arcsinh(j_surf_n / (2.0 * np.maximum(j0_n, 1e-6)))
        eta_p = thermal_v * np.arcsinh(j_surf_p / (2.0 * np.maximum(j0_p, 1e-6)))

        return eta_n, eta_p

    def compute_electrolyte_drop(self, c_e: np.ndarray, T: float) -> float:
        # Concentration overpotential contribution across separator
        # Delta Phi_e ~= 2*(R*T/F)*(1 - t_plus) * ln(c_e(L) / c_e(0))
        c_e_left = max(c_e[0], 1e-3)
        c_e_right = max(c_e[-1], 1e-3)
        
        delta_phi_conc = (2.0 * self.param.R * T / self.param.F) * (1.0 - self.param.t_plus) * np.log(c_e_right / c_e_left)
        return delta_phi_conc

    def calculate_voltage(self, c_e: np.ndarray, c_s_n: np.ndarray, c_s_p: np.ndarray, T: float) -> float:
        u_diff = self.get_ocv_diff(c_s_n[-1], c_s_p[-1])
        eta_n, eta_p = self.compute_reaction_overpotentials(c_e, c_s_n[-1], c_s_p[-1], T)
        delta_phi_e = self.compute_electrolyte_drop(c_e, T)

        # Total voltage under discharge (Iapp > 0 drops potential)
        v_cell = u_diff + eta_p - eta_n + delta_phi_e
        return float(v_cell)


class SEISimulator:
    """
    Solvent reduction kinetics causing SEI growth on negative electrode.
    """

    def __init__(self, param):
        self.param = param

    def step_growth(self, delta_sei: float, eta_sei: float, T: float, dt: float) -> tuple[float, float]:
        # Arrhenius correction for reaction rate
        k_sei = self.param.k_sei_ref * np.exp(
            -(self.param.Ea_sei / self.param.R) * (1.0 / T - 1.0 / 298.15)
        )

        # Diffusion-limited side reaction current
        # Note: added simple resistance term (1 + delta) to avoid infinite current at t=0
        r_film = 1.0 + delta_sei * 1e7
        j_sei = -self.param.F * (k_sei / r_film) * np.exp(
            -0.5 * self.param.F / (self.param.R * T) * eta_sei
        )

        # d(delta)/dt = - (j_sei * M_sei) / (2 * F * rho_sei)
        d_delta_dt = -(j_sei * self.param.M_sei) / (2.0 * self.param.F * self.param.rho_sei)
        
        delta_next = delta_sei + d_delta_dt * dt
        return float(j_sei), float(delta_next)


def verify_mass_balance(geo, c_s_n: np.ndarray, c_s_p: np.ndarray, c_e: np.ndarray, eps_vec: np.ndarray) -> float:
    """
    Validation check: Integrates total moles of Li across solid spheres and electrolyte.
    Should stay practically constant during standard cycles without SEI side loss.
    """
    # Radial integration in active particles (Simpson or trapezoidal approximation)
    mol_vol_n = np.trapz(c_s_n * (geo.r_n ** 2), geo.r_n) * 4.0 * np.pi
    mol_vol_p = np.trapz(c_s_p * (geo.r_p ** 2), geo.r_p) * 4.0 * np.pi

    # Scale by particle volume fraction and electrode dimensions
    n_part_n = (1.0 - geo.eps_n) / ((4.0 / 3.0) * np.pi * (geo.R_n ** 3))
    n_part_p = (1.0 - geo.eps_p) / ((4.0 / 3.0) * np.pi * (geo.R_p ** 3))

    total_li_solid_n = mol_vol_n * n_part_n * geo.L_n
    total_li_solid_p = mol_vol_p * n_part_p * geo.L_p

    # Electrolyte integration across x-grid
    total_li_electrolyte = np.sum(eps_vec * c_e * geo.dx)

    total_moles_li = total_li_solid_n + total_li_solid_p + total_li_electrolyte
    return float(total_moles_li)