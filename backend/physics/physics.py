import math
import numpy as np
from numba import njit
from scipy.optimize import brentq

from backend.physics.config import BatteryGeometry, BatteryParameters, SimulationConfig


@njit(fastmath=True)
def _solid_diffusion_rhs_numba(
    c_s: np.ndarray,
    Dk: float,
    dr: float,
    Rk: float,
    N: int,
    j_int: float,
    F: float
) -> np.ndarray:

    dc_s_dt = np.empty(N)
    flux_faces = np.empty(N + 1)
    flux_faces[0] = 0.0

    # Внутрішні молярні потоки
    for i in range(1, N):
        r_face = i * dr
        flux_faces[i] = -Dk * (r_face * r_face) * (c_s[i] - c_s[i - 1]) / dr

    # Поверхневий потік
    flux_faces[N] = (Rk * Rk) * j_int / F

    # Баланс скінченних об'ємів
    for i in range(N):
        r_low = i * dr
        r_high = (i + 1) * dr
        vol = (r_high**3 - r_low**3) / 3.0
        dc_s_dt[i] = -(flux_faces[i + 1] - flux_faces[i]) / vol

    return dc_s_dt


@njit(fastmath=True)
def _electrolyte_diffusion_rhs_numba(
    c_e: np.ndarray,
    D_e_bulk: float,
    dx: float,
    B_faces: np.ndarray,
    eps: np.ndarray,
    b_x: np.ndarray,
    t_plus: float,
    j_x: np.ndarray,
    N_total: int
) -> np.ndarray:

    dc_e_dt = np.empty(N_total)
    J_full = np.empty(N_total + 1)
    J_full[0] = 0.0
    J_full[N_total] = 0.0

    for i in range(1, N_total):
        J_full[i] = -D_e_bulk * B_faces[i - 1] * (c_e[i] - c_e[i - 1]) / dx

    for i in range(N_total):
        diff = (J_full[i] - J_full[i + 1]) / dx
        source = (1.0 - t_plus) * b_x[i] * j_x[i]
        dc_e_dt[i] = (diff + source) / eps[i]

    return dc_e_dt


@njit(fastmath=True)
def _solve_j_sei_numba(
    U_n: float,
    j0_n: float,
    j_tot: float,
    T: float,
    L_sei: float,
    R: float,
    F: float,
    k_sei_ref: float,
    Ea_k_sei: float,
    T_ref: float,
    D_sol_ref: float,
    Ea_D_sol: float,
    alpha_sei: float,
    c_sol_e: float,
    U_sei: float,
) -> tuple[float, float]:
    k_sei = k_sei_ref * math.exp(-Ea_k_sei / R * (1.0 / T - 1.0 / T_ref))
    D_sol = D_sol_ref * math.exp(-Ea_D_sol / R * (1.0 / T - 1.0 / T_ref))

    j_dl = -F * D_sol * c_sol_e / L_sei

    low = j_dl
    high = 0.0
    mid = 0.0

    for _ in range(50):
        mid = 0.5 * (low + high)
        j_int = j_tot - mid
        
        val = j_int / (2.0 * j0_n)
        eta_int = 2.0 * R * T / F * math.asinh(val)
        eta_sei = U_n + eta_int - U_sei
        
        exponent = -alpha_sei * F * eta_sei / (R * T)
        if exponent > 100.0:
            exponent = 100.0
        elif exponent < -100.0:
            exponent = -100.0
            
        j_rl = -F * k_sei * c_sol_e * math.exp(exponent)
        j_model = (j_rl * j_dl) / (j_rl + j_dl)
        
        if mid - j_model > 0.0:
            high = mid
        else:
            low = mid

    j_int = j_tot - mid
    val = j_int / (2.0 * j0_n)
    eta_int = 2.0 * R * T / F * math.asinh(val)
    eta_sei = U_n + eta_int - U_sei

    return mid, eta_sei


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

    def compute_jk(self, i_app, j_sei=0.0):
        if self.electrode == "n":
            j_tot = i_app / (self.b_k * self.Lk)
            return j_tot - j_sei
        elif self.electrode == "p":
            return -i_app / (self.b_k * self.Lk)
        else:
            raise ValueError("Unknown electrode")

    def compute_derivatives(self, c_s: np.ndarray, i_app: float, T: float, j_sei: float = 0.0) -> np.ndarray:
        p = self.param
        N = len(c_s)

        Dk = self.D_ref * math.exp(
            -self.Ea / p.R * (1.0 / T - 1.0 / p.T_ref)
        )

        dr = self.Rk / N
        j_int = self.compute_jk(i_app, j_sei)

        return _solid_diffusion_rhs_numba(
            c_s=c_s,
            Dk=Dk,
            dr=dr,
            Rk=self.Rk,
            N=N,
            j_int=j_int,
            F=p.F
        )


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

        self.B_faces = (
            2 * self.B_x[:-1] * self.B_x[1:]
            / (self.B_x[:-1] + self.B_x[1:])
        )

    def compute_j_x(self, i_app: float) -> np.ndarray:
        geo = self.geo
        param = self.param

        j_n = np.full(geo.N_x_n, i_app / (geo.L_n * param.F * self.b_x[0]))
        j_s = np.zeros(geo.N_x_s)
        j_p = np.full(geo.N_x_p, -i_app / (geo.L_p * param.F * self.b_x[-1]))

        return np.concatenate([j_n, j_s, j_p])

    def compute_derivatives(self, c_e: np.ndarray, i_app: float, T: float) -> np.ndarray:
        geo = self.geo
        param = self.param

        D_e_bulk = param.D_e_ref * math.exp(
            -(param.Ea_De / param.R) * (1.0 / T - 1.0 / 298.15)
        )

        j_x = self.compute_j_x(i_app)

        return _electrolyte_diffusion_rhs_numba(
            c_e=c_e,
            D_e_bulk=D_e_bulk,
            dx=geo.dx,
            B_faces=self.B_faces,
            eps=self.eps,
            b_x=self.b_x,
            t_plus=param.t_plus,
            j_x=j_x,
            N_total=geo.N_x_total
        )


class TerminalVoltage:

    def __init__(self, geo, param, sim):
        self.param, self.geo, self.sim = param, geo, sim
        self._b_n = param.get_b_n(geo)
        self._b_p = param.get_b_p(geo)

    def Ueq(self, c_s_n, c_s_p):
        param = self.param
        th_n = c_s_n[-1] / param.c_n_max
        th_p = c_s_p[-1] / param.c_p_max
        return param.OCV_p(th_p) - param.OCV_n(th_n)

    def compute_j0_x(self, c_e_slice, c_k_surf, K_k, c_k_max):
        param = self.param
        radicand = ((c_e_slice / param.c_e_0) * (c_k_surf / c_k_max) * (1 - c_k_surf / c_k_max))
        return (param.F * K_k * np.sqrt(radicand)).mean()

    def eta_r(self, c_e, c_s_n, c_s_p, i_app, T, j_sei=0.0):
        geo = self.geo
        param = self.param
        
        j_tot_n = i_app / (self._b_n * geo.L_n)
        j_n = j_tot_n - j_sei
        j_p = -i_app / (self._b_p * geo.L_p)

        j_n0 = self.compute_j0_x(
            c_e[:geo.N_x_n], c_s_n[-1], param.K_n, param.c_n_max
        )
        j_p0 = self.compute_j0_x(
            c_e[geo.N_x_n + geo.N_x_s:], c_s_p[-1], param.K_p, param.c_p_max
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
            c_e[:g.N_x_n].mean() - c_e[g.N_x_n + g.N_x_s:].mean()
        )

    def dphi_e(self, i_app): 
        return i_app / self.param.sigma_e * (
            self.geo.L_n / (3 * self.geo.B_n) + self.geo.L_s / self.geo.B_s + self.geo.L_p / (3 * self.geo.B_p)
        )
    
    def dphi_s(self, i_app): 
        return i_app / 3 * (
            self.geo.L_p / self.param.sigma_p + self.geo.L_n / self.param.sigma_n
        )

    def dphi_f(self, i_app, L_sei: float) -> float:
        return (
            i_app * L_sei
            / (self.geo.L_n * self._b_n * self.param.sigma_sei)
        )

    def calculate(self, c_s_n, c_s_p, c_e, i_app, T, L_sei, j_sei=0):
        U = (
            self.Ueq(c_s_n, c_s_p)
            - self.eta_r(c_e, c_s_n, c_s_p, i_app, T, j_sei)
            - self.eta_c(c_e, T)
            - self.dphi_e(i_app)
            - self.dphi_s(i_app)
            - self.dphi_f(i_app, L_sei)
        )
        return U


class ThermalModel:

    def __init__(self, geo: BatteryGeometry, param: BatteryParameters, sim: SimulationConfig):
        self.geo = geo
        self.param = param
        self.sim = sim

    def compute_Q_irr(self, I: float, Ueq: float, U: float) -> float:
        return I * (Ueq - U)

    def compute_Q_rev(self, I: float, T: float) -> float:
        return -I * T * self.param.dUeq_dT

    def compute_Q_cooling(self, T: float) -> float:
        return (
            self.param.h_cool
            * self.geo.A_surface
            * (T - self.param.T_amb)
        )
    
    def compute_derivatives(self, T: float, Ueq: float, U: float, I: float) -> float:
        p = self.param
        g = self.geo

        Q_irr = self.compute_Q_irr(I, Ueq, U)
        Q_rev = self.compute_Q_rev(I, T)
        Q_cooling = self.compute_Q_cooling(T)

        C_thermal = p.get_theta_thermal(g) * g.V_batt
        dT_dt = (Q_irr + Q_rev - Q_cooling) / C_thermal

        return float(dT_dt)


class SEIGrowth:

    def __init__(self, param):
        self.param = param

    def compute_k_sei(self, T: float) -> float:
        p = self.param
        return p.k_sei_ref * math.exp(-p.Ea_k_sei / p.R * (1.0 / T - 1.0 / p.T_ref))

    def compute_D_sol(self, T: float) -> float:
        p = self.param
        return p.D_sol_ref * math.exp(-p.Ea_D_sol / p.R * (1.0 / T - 1.0 / p.T_ref))

    def compute_eta_sei(self, U_n: float, j_int: float, j0_n: float, T: float) -> float:
        p = self.param
        if j0_n <= 0:
            raise ValueError("j0_n must be positive")
        eta_int = 2.0 * p.R * T / p.F * math.asinh(j_int / (2.0 * j0_n))
        return U_n + eta_int - p.U_sei

    def compute_j_rl(self, T: float, eta_sei: float) -> float:
        p = self.param
        k_sei = self.compute_k_sei(T)
        exponent = -p.alpha_sei * p.F * eta_sei / (p.R * T)
        if exponent > 100.0: exponent = 100.0
        elif exponent < -100.0: exponent = -100.0
        return -p.F * k_sei * p.c_sol_e * math.exp(exponent)

    def compute_j_dl(self, T: float, L_sei: float) -> float:
        p = self.param
        if L_sei <= 0:
            raise ValueError("L_sei must be positive")
        D_sol = self.compute_D_sol(T)
        return -p.F * D_sol * p.c_sol_e / L_sei

    def compute_j_sei(self, T: float, eta_sei: float, L_sei: float) -> float:
        j_rl = self.compute_j_rl(T, eta_sei)
        j_dl = self.compute_j_dl(T, L_sei)
        return j_rl * j_dl / (j_rl + j_dl)

    def solve_j_sei(self, U_n: float, j0_n: float, j_tot: float, T: float, L_sei: float) -> tuple[float, float]:
        p = self.param
        return _solve_j_sei_numba(
            U_n=float(U_n),
            j0_n=float(j0_n),
            j_tot=float(j_tot),
            T=float(T),
            L_sei=float(L_sei),
            R=float(p.R),
            F=float(p.F),
            k_sei_ref=float(p.k_sei_ref),
            Ea_k_sei=float(p.Ea_k_sei),
            T_ref=float(p.T_ref),
            D_sol_ref=float(p.D_sol_ref),
            Ea_D_sol=float(p.Ea_D_sol),
            alpha_sei=float(p.alpha_sei),
            c_sol_e=float(p.c_sol_e),
            U_sei=float(p.U_sei),
        )

    def compute_derivatives(self, j_sei: float, A_active: float) -> tuple[float, float]:
        p = self.param
        dL_sei_dt = -p.M_sei * j_sei / (p.n_sei * p.F * p.rho_sei)
        dQ_LLI_dt = -A_active * j_sei
        return float(dL_sei_dt), float(dQ_LLI_dt)