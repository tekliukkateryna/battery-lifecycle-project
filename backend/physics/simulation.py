import numpy as np
from scipy.integrate import solve_ivp

from backend.physics.physics import (
    SolidDiffusion,
    ElectrolyteDiffusionSPMe,
    TerminalVoltage,
    ThermalModel,
    SEIGrowth,
)


class BatterySimulation:
    def __init__(
        self,
        geo,
        param,
        sim,
        enable_sei=False,
        enable_thermal=False,
        include_reversible_heat=False,
    ):
        self.geo = geo
        self.param = param
        self.sim = sim
        self.enable_sei = enable_sei
        self.enable_thermal = enable_thermal
        self.include_reversible_heat = include_reversible_heat

        self.solid_n = SolidDiffusion(geo, param, sim, "n")
        self.solid_p = SolidDiffusion(geo, param, sim, "p")
        self.electrolyte = ElectrolyteDiffusionSPMe(geo, param, sim)
        self.voltage = TerminalVoltage(geo, param, sim)
        self.thermal = ThermalModel(param, sim)
        self.sei = SEIGrowth(param)

        n_n, n_p, n_e = geo.N_r_n, geo.N_r_p, geo.N_x_total
        self.s_n = slice(0, n_n)
        self.s_p = slice(n_n, n_n + n_p)
        self.s_e = slice(n_n + n_p, n_n + n_p + n_e)
        self.i_T = n_n + n_p + n_e
        self.i_L = self.i_T + 1
        self.i_Q = self.i_T + 2
        self.n_states = self.i_Q + 1

        # States have very different units/scales; protect the nanometre SEI state.
        self.atol = np.full(self.n_states, 1e-5)
        self.atol[self.i_T] = 1e-6
        self.atol[self.i_L] = 1e-13
        self.atol[self.i_Q] = 1e-9

    def initial_state(self):

        y0 = np.empty(self.n_states)

        # Initial solid concentrations [mol/m³]
        c_n_init, c_p_init = (
            self.sim.initial_solid_concentrations(self.param)
        )

        y0[self.s_n] = c_n_init
        y0[self.s_p] = c_p_init

        # Electrolyte concentration [mol/m³]
        y0[self.s_e] = self.sim.c_e_init

        # Temperature [K]
        y0[self.i_T] = self.sim.T_init

        # SEI thickness [m]
        y0[self.i_L] = self.param.L_sei_init

        # Initial lithium inventory loss [C]
        y0[self.i_Q] = 0.0

        return y0

    def unpack(self, y):
        return (
            y[self.s_n], y[self.s_p], y[self.s_e],
            float(y[self.i_T]), float(y[self.i_L]), float(y[self.i_Q]),
        )

    def sei_current(self, c_s_n, c_e, T, L_sei):
        """Return (j_sei [A/m^2 active area], eta_sei [V])."""
        if not self.enable_sei:
            return 0.0, 0.0

        p, g, sim = self.param, self.geo, self.sim
        theta_n = c_s_n[-1] / p.c_n_max
        U_n = sim.OCV_n(theta_n)
        j0_n = self.voltage.compute_j0_x(
            c_e[:g.N_x_n], c_s_n[-1], p.K_n, p.c_n_max
        )
        j_tot_n = sim.i_app / (p.get_b_n(g) * g.L_n)

        return self.sei.solve_j_sei(
            U_n=U_n,
            j0_n=j0_n,
            j_tot=j_tot_n,
            T=T,
            L_sei=L_sei,
        )

    def rhs(self, t, y):
        c_s_n, c_s_p, c_e, T, L_sei, _ = self.unpack(y)

        j_sei, _ = self.sei_current(c_s_n, c_e, T, L_sei)

        dy = np.zeros_like(y)
        dy[self.s_n] = self.solid_n.compute_derivatives(c_s_n, T, j_sei)
        dy[self.s_p] = self.solid_p.compute_derivatives(c_s_p, T)
        dy[self.s_e] = self.electrolyte.compute_derivatives(c_e, T)

        # SEI film growth and lithium inventory loss
        if self.enable_sei:
            a_n = self.param.get_b_n(self.geo)
            active_area = self.sim.A_cell * a_n * self.geo.L_n
            dL_dt, dQ_dt = self.sei.compute_derivatives(j_sei, active_area)
            dy[self.i_L] = dL_dt
            dy[self.i_Q] = dQ_dt

        # Isothermal mode: retain temperature as a state but dT/dt=0.
        if self.enable_thermal:
            film_thickness_for_voltage = L_sei if self.enable_sei else 0.0
            Ueq = self.voltage.Ueq(c_s_n, c_s_p)
            V = self.voltage.calculate(
                c_s_n, c_s_p, c_e, T,
                film_thickness_for_voltage, j_sei
            )
            I_app = self.sim.i_app * self.sim.A_cell  # [A]

            if self.include_reversible_heat:
                dy[self.i_T] = self.thermal.compute_derivatives(
                    T, Ueq, V, I_app, c_s_n, c_s_p
                )
            else:
                Q_irr = self.thermal.compute_Q_irr(I_app, Ueq, V)
                Q_cool = self.thermal.compute_Q_cooling(T)
                dy[self.i_T] = (
                    Q_irr - Q_cool
                ) / (self.param.mass * self.param.Cp)

        return dy

    def run(self, t_final=None, num_points=201):
        end = float(self.sim.t_final if t_final is None else t_final)
        if end <= 0:
            raise ValueError("t_final must be positive")
        times = np.linspace(0.0, end, num_points)
        sol = solve_ivp(
            self.rhs, (0.0, end), self.initial_state(),
            method="BDF", t_eval=times,
            rtol=1e-6, atol=self.atol,
            max_step=60.0,
        )
        if not sol.success:
            raise RuntimeError(f"Battery integration failed: {sol.message}")
        return sol

    def diagnostics(self, sol):
        voltage = np.empty(sol.t.size)
        j_sei_values = np.empty(sol.t.size)
        for i in range(sol.t.size):
            c_s_n, c_s_p, c_e, T, L_sei, _ = self.unpack(sol.y[:, i])
            j_sei, _ = self.sei_current(c_s_n, c_e, T, L_sei)
            voltage[i] = self.voltage.calculate(
                c_s_n, c_s_p, c_e, T,
                L_sei if self.enable_sei else 0.0,
                j_sei,
            )
            j_sei_values[i] = j_sei
        return {
            "time_s": sol.t,
            "voltage_V": voltage,
            "temperature_K": sol.y[self.i_T],
            "sei_thickness_m": sol.y[self.i_L],
            "capacity_loss_Ah": sol.y[self.i_Q] / 3600.0,
            "j_sei_A_m2": j_sei_values,
        }
