import numpy as np
from scipy.optimize import brentq

from backend.physics.config import BatteryGeometry, BatteryParameters, SimulationConfig

class SolidDiffusion:

    def __init__(self, geo: BatteryGeometry, param: BatteryParameters, sim: SimulationConfig, electrode):
        self.geo = geo
        self.param = param
        self.electrode = electrode
        self.sim = sim

        if electrode == "n":
            r = geo.r_n
            Rk = geo.R_n
            Lk = geo.L_n
            D_ref = param.D_s_n_ref
            Ea = param.Ea_Ds_n
            b_k = param.get_b_n(geo)
            sign = 1.0

        elif electrode == "p":
            r = geo.r_p
            Rk = geo.R_p
            Lk = geo.L_p
            D_ref = param.D_s_p_ref
            Ea = param.Ea_Ds_p
            b_k = param.get_b_p(geo)
            sign = -1.0

        else:
            raise ValueError("electrode must be 'n' or 'p'")

        self.r = r
        self.Rk = Rk
        self.Lk = Lk
        self.D_ref = D_ref
        self.Ea = Ea
        self.b_k = b_k
        self.sign = sign

    def compute_jk(self, j_sei=0.0):
        p = self.param

        if self.electrode == "n":
            j_tot = (
                self.sim.i_app
                / (self.b_k * self.Lk)
            )

            return j_tot - j_sei

        elif self.electrode == "p":
            return (
                -self.sim.i_app
                / (self.b_k * self.Lk)
            )

        else:
            raise ValueError("Unknown electrode")

    def compute_derivatives(
        self,
        c_s: np.ndarray,
        T: float,
        j_sei: float = 0.0
    ) -> np.ndarray:

        param = self.param

        dc_s_dt = np.zeros_like(c_s)

        dr = self.r[1] - self.r[0]

        # Temperature-dependent diffusion coefficient
        Dk = self.D_ref * np.exp(
            -(self.Ea / param.R)
            * (1.0 / T - 1.0 / 298.15)
        )

        # Interior nodes
        d2c_dr2 = (
            c_s[2:] - 2 * c_s[1:-1] + c_s[:-2]
        ) / dr**2

        dc_dr = (
            c_s[2:] - c_s[:-2]
        ) / (2 * dr)

        dc_s_dt[1:-1] = Dk * (
            d2c_dr2
            + (2 / self.r[1:-1]) * dc_dr
        )

        # Center boundary condition
        dc_s_dt[0] = (
            6 * Dk * (c_s[1] - c_s[0]) / dr**2
        )

        # Intercalation current density [A/m²]
        j_int = self.compute_jk(j_sei)

        # Molar surface flux [mol/(m² s)]
        surface_flux = j_int / param.F

        # Surface boundary condition
        dc_s_dt[-1] = (
            (2 * Dk / dr**2)
            * (c_s[-2] - c_s[-1])
            - (2 / dr) * surface_flux
            - (2 / self.Rk) * surface_flux
        )

        return dc_s_dt

class ElectrolyteDiffusionSPMe:

    def __init__(self, geo: BatteryGeometry, param: BatteryParameters, sim: SimulationConfig):

        self.geo = geo
        self.param = param
        self.sim = sim

        self.eps = np.concatenate([
            np.full(geo.N_x_n, geo.eps_n),
            np.full(geo.N_x_s, geo.eps_s),
            np.full(geo.N_x_p, geo.eps_p),
        ])

        self.B_x = np.concatenate([
            np.full(geo.N_x_n, geo.B_n),
            np.full(geo.N_x_s, geo.B_s),
            np.full(geo.N_x_p, geo.B_p),
        ])

        self.b_x = np.concatenate([
            np.full(geo.N_x_n, param.get_b_n(geo)),
            np.full(geo.N_x_s, param.get_b_s(geo)),
            np.full(geo.N_x_p, param.get_b_p(geo)),
        ])

        # Harmonic mean at internal cell faces
        self.B_faces = (
            2 * self.B_x[:-1] * self.B_x[1:]
            / (self.B_x[:-1] + self.B_x[1:])
        )

    def compute_j_x(self, i_app: float) -> np.ndarray:
        geo = self.geo
        param = self.param

        # Molar interfacial flux [mol/(m² s)]
        j_n = np.full(
            geo.N_x_n,
            i_app / (geo.L_n * param.F * param.get_b_n(geo))
        )

        j_s = np.zeros(geo.N_x_s)

        j_p = np.full(
            geo.N_x_p,
            -i_app / (geo.L_p * param.F * param.get_b_p(geo))
        )

        return np.concatenate([j_n, j_s, j_p])

    def compute_derivatives(self, c_e: np.ndarray, T: float) -> np.ndarray:

        geo = self.geo
        param = self.param

        D_e_bulk = param.D_e_ref * np.exp(
            -(param.Ea_De / param.R)
            * (1 / T - 1 / 298.15)
        )

        j_x = self.compute_j_x(self.sim.i_app)

        # Face diffusive flux, positive in +x direction
        J_faces = (
            -D_e_bulk * self.B_faces
            * (c_e[1:] - c_e[:-1]) / geo.dx
        )

        # Zero flux at the external boundaries
        J_full = np.pad(J_faces, (1, 1), mode="constant", constant_values=0)

        # Divergence of diffusive flux
        diffusion = (
            J_full[:-1] - J_full[1:]
        ) / geo.dx

        # Reaction source
        source = (
            (1 - param.t_plus) * self.b_x * j_x
        )

        dc_e_dt = (diffusion + source) / self.eps

        return dc_e_dt


class TerminalVoltage:
    def __init__(self, geo, param, sim):
        self.param, self.geo, self.sim = param, geo, sim

    def Ueq(self, c_s_n, c_s_p):
        param = self.param
        sim = self.sim
        th_n = c_s_n[-1] / param.c_n_max
        th_p = c_s_p[-1] / param.c_p_max
        return sim.OCV_p(th_p) - sim.OCV_n(th_n)

    def compute_j0_x(self, c_e_slice, c_k_surf, K_k, c_k_max):

        param = self.param
        
        radicand = ((c_e_slice / param.c_e_0) * (c_k_surf/c_k_max) * (1 - c_k_surf/c_k_max))
        return (param.F * K_k * np.sqrt(radicand)).mean()                     # dimensional form, [A/m²]

    def eta_r(self, c_e, c_s_n, c_s_p, T, j_sei=0.0):
        
        geo = self.geo
        param = self.param
        sim = self.sim
        
        # Total negative electrode current density [A/m²]
        j_tot_n = (
            sim.i_app
            / (param.get_b_n(geo) * geo.L_n)
            )
        
        # Intercalation current density [A/m²]
        j_n = j_tot_n - j_sei
        
        # Positive electrode current density [A/m²]
        j_p = (
            -sim.i_app
            / (param.get_b_p(geo) * geo.L_p)
        )

        # Exchange current densities [A/m²]
        j_n0 = self.compute_j0_x(
            c_e[:geo.N_x_n],
            c_s_n[-1],
            param.K_n,
            param.c_n_max
        )

        j_p0 = self.compute_j0_x(
            c_e[geo.N_x_n + geo.N_x_s:],
            c_s_p[-1],
            param.K_p,
            param.c_p_max
        )

        return (
            2 * param.R * T / param.F
            * (
                np.arcsinh(j_n / (2 * j_n0))
                - np.arcsinh(j_p / (2 * j_p0))
            )
        )

    def eta_c(self, c_e, T):
        g, p = self.geo, self.param
        return (1 - p.t_plus) * 2 * p.R * T / (p.F * p.c_e_0) * (
            c_e[:g.N_x_n].mean() - c_e[g.N_x_n + g.N_x_s:].mean())

    def dphi_e(self): 
        
        return self.sim.i_app / self.param.sigma_e * (
            self.geo.L_n / (3 * self.geo.B_n) + self.geo.L_s / self.geo.B_s + self.geo.L_p / (3 * self.geo.B_p))
    
    def dphi_s(self): 
        return self.sim.i_app / 3 * (
            self.geo.L_p / self.param.sigma_p + self.geo.L_n / self.param.sigma_n)

    def dphi_f(self, L_sei: float) -> float:
        p = self.param
        geo = self.geo
        sim = self.sim
        
        a_n = p.get_b_n(geo)
        
        # Film resistance contribution [V]
        return (
            sim.i_app * L_sei
            / (geo.L_n * a_n * p.sigma_sei)
            )


    def calculate(
        self,
        c_s_n,
        c_s_p,
        c_e,
        T,
        L_sei,
        j_sei=0.0
    ):

        U = (
            self.Ueq(c_s_n, c_s_p)
            - self.eta_r(c_e, c_s_n, c_s_p, T, j_sei)
            - self.eta_c(c_e, T)
            - self.dphi_e()
            - self.dphi_s()
            - self.dphi_f(L_sei)
        )

        return U

class ThermalModel:

    def __init__(self, param: BatteryParameters, sim: SimulationConfig):

        self.param = param
        self.sim = sim

        # Experimental entropy coefficient tables
        # To be populated from literature data
        self.theta_n_data = None
        self.dUn_dT_data = None

        self.theta_p_data = None
        self.dUp_dT_data = None

    def dOCV_n_dT(self, theta_n):
        if self.theta_n_data is None or self.dUn_dT_data is None:
            raise ValueError(
                "Negative electrode entropy data are not defined"
            )

        return np.interp(
            theta_n,
            self.theta_n_data,
            self.dUn_dT_data
        )

    def dOCV_p_dT(self, theta_p):
        if self.theta_p_data is None or self.dUp_dT_data is None:
            raise ValueError(
                "Positive electrode entropy data are not defined"
            )

        return np.interp(
            theta_p,
            self.theta_p_data,
            self.dUp_dT_data
        )

    def compute_Q_irr(self, I: float, Ueq: float, U: float) -> float:

        return I * (Ueq - U)

    def compute_Q_rev(self, I: float, T: float, dUeq_dT: float) -> float:
        
        return -I * T * dUeq_dT

    def compute_Q_cooling(self, T: float) -> float:
        
        p = self.param

        return (
            p.h_cool
            * p.A_surface
            * (T - p.T_amb)
        )

    def compute_derivatives(self, T: float, Ueq: float, U: float, I: float, c_s_n: np.ndarray, c_s_p: np.ndarray) -> float:

        p = self.param

        # Electrode surface stoichiometries
        theta_n = c_s_n[-1] / p.c_n_max
        theta_p = c_s_p[-1] / p.c_p_max

        # Full-cell entropic coefficient [V/K]
        '''dUeq_dT = (
            self.dOCV_p_dT(theta_p)
            - self.dOCV_n_dT(theta_n)
        )'''
        dUeq_dT = 0

        # Heat generation [W]
        Q_irr = self.compute_Q_irr(I, Ueq, U)

        Q_rev = self.compute_Q_rev(
            I, T, dUeq_dT
        )

        # Cooling [W]
        Q_cooling = self.compute_Q_cooling(T)

        # Total heat capacity [J/K]
        C_thermal = p.mass * p.Cp

        # Temperature derivative [K/s]
        dT_dt = (
            Q_irr + Q_rev - Q_cooling
        ) / C_thermal

        return float(dT_dt)


# Поки не враховує зміну пористості в elecrolyte diffusion в зоні аноду
class SEIGrowth:

    def __init__(self, param):
        self.param = param


    def compute_k_sei(self, T: float) -> float:
        p = self.param

        return p.k_sei_ref * np.exp(
            -p.Ea_k_sei / p.R
            * (1.0 / T - 1.0 / p.T_ref)
        )

    def compute_D_sol(self, T: float) -> float:
        p = self.param

        return p.D_sol_ref * np.exp(
            -p.Ea_D_sol / p.R
            * (1.0 / T - 1.0 / p.T_ref)
        )


    def compute_eta_sei(
        self,
        U_n: float,
        j_int: float,
        j0_n: float,
        T: float
    ) -> float:

        p = self.param

        if j0_n <= 0:
            raise ValueError("j0_n must be positive")

        eta_int = (
            2.0 * p.R * T / p.F
            * np.arcsinh(j_int / (2.0 * j0_n))
        )

        return U_n + eta_int - p.U_sei

    # Reaction-limited SEI current
    def compute_j_rl(
        self,
        T: float,
        eta_sei: float
    ) -> float:

        p = self.param

        k_sei = self.compute_k_sei(T)

        exponent = (
            -p.alpha_sei * p.F * eta_sei
            / (p.R * T)
        )

        # Numerical protection against overflow
        exponent = np.clip(exponent, -100.0, 100.0)

        return (
            -p.F * k_sei * p.c_sol_e
            * np.exp(exponent)
        )

    # Diffusion-limited SEI current
    def compute_j_dl(
        self,
        T: float,
        L_sei: float
    ) -> float:

        p = self.param

        if L_sei <= 0:
            raise ValueError("L_sei must be positive")

        D_sol = self.compute_D_sol(T)

        return (
            -p.F * D_sol * p.c_sol_e
            / L_sei
        )

    # Combined SEI current
    def compute_j_sei(
        self,
        T: float,
        eta_sei: float,
        L_sei: float
    ) -> float:

        j_rl = self.compute_j_rl(T, eta_sei)
        j_dl = self.compute_j_dl(T, L_sei)

        return j_rl * j_dl / (j_rl + j_dl)

    # Self-consistent SEI current
    def solve_j_sei(
        self,
        U_n: float,
        j0_n: float,
        j_tot: float,
        T: float,
        L_sei: float
    ) -> tuple[float, float]:

        j_dl = self.compute_j_dl(T, L_sei)

        def residual(j_sei):

            # Intercalation current [A/m²]
            j_int = j_tot - j_sei

            # SEI overpotential [V]
            eta_sei = self.compute_eta_sei(
                U_n, j_int, j0_n, T
            )

            # SEI current [A/m²]
            j_model = self.compute_j_sei(
                T, eta_sei, L_sei
            )

            return j_sei - j_model

        # Both limiting currents are negative.
        # Physical root lies between j_dl and zero.
        j_sei = brentq(
            residual,
            j_dl,
            0.0,
            xtol=1e-20,
            rtol=1e-12
        )

        j_int = j_tot - j_sei

        eta_sei = self.compute_eta_sei(
            U_n, j_int, j0_n, T
        )

        return float(j_sei), float(eta_sei)

    # SEI growth and lithium inventory loss
    def compute_derivatives(
        self,
        j_sei: float,
        A_active: float
    ) -> tuple[float, float]:

        p = self.param

        # SEI thickness growth [m/s]
        dL_sei_dt = (
            -p.M_sei * j_sei
            / (p.n_sei * p.F * p.rho_sei)
        )

        # Loss of cyclable lithium inventory [C/s]
        dQ_LLI_dt = -A_active * j_sei

        return float(dL_sei_dt), float(dQ_LLI_dt)