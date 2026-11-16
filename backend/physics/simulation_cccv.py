import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
from scipy.optimize import brentq

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
        enable_sei,
        enable_thermal
    ):
        self.geo = geo
        self.param = param
        self.sim = sim
        self.enable_sei = enable_sei
        self.enable_thermal = enable_thermal

        self.solid_n = SolidDiffusion(geo, param, sim, "n")
        self.solid_p = SolidDiffusion(geo, param, sim, "p")
        self.electrolyte = ElectrolyteDiffusionSPMe(geo, param, sim)
        self.voltage = TerminalVoltage(geo, param, sim)
        self.thermal = ThermalModel(geo, param, sim)
        self.sei = SEIGrowth(param)

        n_n, n_p, n_e = geo.N_r_n, geo.N_r_p, geo.N_x_total
        self.s_n = slice(0, n_n)
        self.s_p = slice(n_n, n_n + n_p)
        self.s_e = slice(n_n + n_p, n_n + n_p + n_e)
        self.i_T = n_n + n_p + n_e
        self.i_L = self.i_T + 1
        self.i_Q = self.i_T + 2
        self.n_states = self.i_Q + 1

        self._b_n = param.get_b_n(geo)

        self.atol = np.full(self.n_states, 1e-5)
        self.atol[self.i_T] = 1e-6
        self.atol[self.i_L] = 1e-13
        self.atol[self.i_Q] = 1e-9

        # Побудова та кешування спарсивної маски Якобіана
        self.jac_sparsity = self._build_jac_sparsity()

        # State continues across successive run_cycles() calls.
        self._last_state = None
        self._last_time = 0.0

        # Applied-current controller: CC uses sim.I_app; CV solves for I(y).
        self._control_mode = "CC"
        self._cv_voltage = None
        self._cv_max_current = None

    def _build_jac_sparsity(self):
        n = self.n_states
        J = lil_matrix((n, n), dtype=int)

        n_n = self.geo.N_r_n
        for i in range(n_n):
            J[i, i] = 1
            if i > 0: J[i, i - 1] = 1
            if i < n_n - 1: J[i, i + 1] = 1
            J[i, self.i_T] = 1

        for col in range(self.geo.N_x_n):
            J[n_n - 1, self.s_e.start + col] = 1
        J[n_n - 1, self.i_L] = 1

        n_p = self.geo.N_r_p
        off_p = self.s_p.start
        for i in range(n_p):
            idx = off_p + i
            J[idx, idx] = 1
            if i > 0: J[idx, idx - 1] = 1
            if i < n_p - 1: J[idx, idx + 1] = 1
            J[idx, self.i_T] = 1

        n_e = self.geo.N_x_total
        off_e = self.s_e.start
        for i in range(n_e):
            idx = off_e + i
            J[idx, idx] = 1
            if i > 0: J[idx, idx - 1] = 1
            if i < n_e - 1: J[idx, idx + 1] = 1
            J[idx, self.i_T] = 1


        J[self.i_T, self.s_n.start + n_n - 1] = 1
        J[self.i_T, self.s_p.start + n_p - 1] = 1
        for col in range(n_e):
            J[self.i_T, off_e + col] = 1
        J[self.i_T, self.i_T] = 1
        J[self.i_T, self.i_L] = 1

        for row in (self.i_L, self.i_Q):
            J[row, self.s_n.start + n_n - 1] = 1
            for col in range(self.geo.N_x_n):
                J[row, off_e + col] = 1
            J[row, self.i_T] = 1
            J[row, self.i_L] = 1

        return J.tocsc()

    @property
    def i_app(self):
        return self.sim.I_app / self.geo.A_cell

    def initial_state(self):
        y0 = np.empty(self.n_states)
        c_n_init, c_p_init = (
            self.sim.initial_solid_concentrations(self.param)
        )
        y0[self.s_n] = c_n_init
        y0[self.s_p] = c_p_init
        y0[self.s_e] = self.sim.c_e_init
        y0[self.i_T] = self.sim.T_init
        y0[self.i_L] = self.param.L_sei_init
        y0[self.i_Q] = 0.0
        return y0

    def _unpack(self, y: np.ndarray):
        return (
            y[self.s_n], y[self.s_p], y[self.s_e],
            float(y[self.i_T]), float(y[self.i_L]), float(y[self.i_Q]),
        )

    def _sei_current(self, c_s_n: np.ndarray, c_e: np.ndarray, T: float, L_sei: float):
        if not self.enable_sei:
            return 0.0

        return self.sei.compute_j_dl(T, L_sei)

    def _current_for_state(self, y):
        """Actual terminal current in A: negative for charge, positive for discharge."""
        if self._control_mode != "CV":
            return float(self.sim.I_app)

        c_s_n, c_s_p, c_e, T, L_sei, _ = self._unpack(y)
        j_sei = self._sei_current(c_s_n, c_e, T, L_sei)
        L_film = L_sei if self.enable_sei else 0.0

        def voltage_error(I):
            return float(self.voltage.calculate(
                c_s_n, c_s_p, c_e,
                I / self.geo.A_cell,
                T, L_film, j_sei,
            )) - self._cv_voltage

        low = -self._cv_max_current  # Maximum allowed charging current
        high = 0.0                  # No positive (discharging) current in CV
        f_low = voltage_error(low)
        f_high = voltage_error(high)

        # Current saturation is only relevant very close to a switching event
        # or on solver trial states. In the interior, brentq enforces V=Vmax.
        if f_low <= 0.0:
            return low
        if f_high >= 0.0:
            return high

        return float(brentq(voltage_error, low, high, xtol=1e-10))

    def _temperature_derivative(self, c_s_n, c_s_p, c_e, T, L_sei, j_sei, I):
        if not self.enable_thermal:
            return 0.0

        L_film = L_sei if self.enable_sei else 0.0
        Ueq = self.voltage.Ueq(c_s_n, c_s_p)
        V = self.voltage.calculate(
            c_s_n, c_s_p, c_e,
            I / self.geo.A_cell, T, L_film, j_sei,
        )
        return self.thermal.compute_derivatives(T, Ueq, V, I)

    def _rhs(self, t, y):
        c_s_n, c_s_p, c_e, T, L_sei, _ = self._unpack(y)
        j_sei = self._sei_current(c_s_n, c_e, T, L_sei)
        I = self._current_for_state(y)
        i_app = I / self.geo.A_cell

        dy = np.zeros_like(y)
        dy[self.s_n] = self.solid_n.compute_derivatives(c_s_n, i_app, T, j_sei)
        dy[self.s_p] = self.solid_p.compute_derivatives(c_s_p, i_app, T)
        dy[self.s_e] = self.electrolyte.compute_derivatives(c_e, i_app, T)

        if self.enable_sei:
            active_area = self.geo.A_cell * self._b_n * self.geo.L_n
            dL_dt, dQ_dt = self.sei.compute_derivatives(j_sei, active_area)
            dy[self.i_L] = dL_dt
            dy[self.i_Q] = dQ_dt

        if self.enable_thermal:
            dy[self.i_T] = self._temperature_derivative(
                c_s_n, c_s_p, c_e, T, L_sei, j_sei, I,
            )
        return dy

    def _terminal_voltage(self, y):
        c_s_n, c_s_p, c_e, T, L_sei, _ = self._unpack(y)
        j_sei = self._sei_current(c_s_n, c_e, T, L_sei)
        I = self._current_for_state(y)
        return float(self.voltage.calculate(
            c_s_n, c_s_p, c_e, I / self.geo.A_cell,
            T, L_sei if self.enable_sei else 0.0, j_sei,
        ))

    def _integrate_interval(
        self,
        t_start,
        t_end,
        y0,
        num_points,
        voltage_cutoff=None,
        direction=0,
        current_cutoff=None,
    ):
        if t_end <= t_start:
            raise ValueError("Invalid time interval")
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        if voltage_cutoff is not None and current_cutoff is not None:
            raise ValueError("Choose a voltage or current cutoff, not both")

        times = np.linspace(t_start, t_end, num_points)
        event = None

        if voltage_cutoff is not None:
            def voltage_event(t, y):
                return self._terminal_voltage(y) - voltage_cutoff
            voltage_event.terminal = True
            voltage_event.direction = direction
            event = voltage_event

        if current_cutoff is not None:
            # Charging current is negative: stop when I grows to -I_cutoff.
            def current_event(t, y):
                return self._current_for_state(y) + current_cutoff
            current_event.terminal = True
            current_event.direction = +1
            event = current_event

        sol = solve_ivp(
            self._rhs,
            (t_start, t_end),
            y0,
            method="BDF",
            t_eval=times,
            events=event,
            rtol=1e-6,
            atol=self.atol,
            max_step=60.0,
            # In CV, current depends on every state in the voltage model.
            # The sparse CC Jacobian mask is not valid for this dense coupling.
            jac_sparsity=None if self._control_mode == "CV" else self.jac_sparsity,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        if event is not None:
            if sol.t_events[0].size == 0:
                which = (f"Voltage cutoff {voltage_cutoff} V" if voltage_cutoff is not None
                         else f"CV current cutoff {current_cutoff} A")
                raise RuntimeError(
                    f"{which} not reached by t={t_end:.1f} s; "
                    "increase the corresponding phase time limit."
                )

            t_hit = float(sol.t_events[0][-1])
            y_hit = sol.y_events[0][-1].copy()
            if sol.t.size == 0 or not np.isclose(sol.t[-1], t_hit, atol=1e-9, rtol=0.0):
                sol.t = np.append(sol.t, t_hit)
                sol.y = np.column_stack((sol.y, y_hit))
            else:
                sol.t[-1] = t_hit
                sol.y[:, -1] = y_hit

        return sol

    def run(self, t_final=None, num_points=201):
        end = float(
            self.sim.t_final if t_final is None else t_final
        )

        if end <= 0:
            raise ValueError("t_final must be positive")

        return self._integrate_interval(
            t_start=0.0,
            t_end=end,
            y0=self.initial_state(),
            num_points=num_points,
        )

    def diagnostics(self, sol):
        voltage = np.empty(sol.t.size)
        j_sei_values = np.empty(sol.t.size)
        current_values = np.empty(sol.t.size)
        for i in range(sol.t.size):
            y = sol.y[:, i]
            c_s_n, c_s_p, c_e, T, L_sei, _ = self._unpack(y)
            j_sei = self._sei_current(c_s_n, c_e, T, L_sei)
            I = self._current_for_state(y)
            current_values[i] = I
            voltage[i] = self.voltage.calculate(
                c_s_n, c_s_p, c_e, I / self.geo.A_cell, T,
                L_sei if self.enable_sei else 0.0, j_sei,
            )
            j_sei_values[i] = j_sei
        return {
            "time_s": sol.t,
            "voltage_V": voltage,
            "temperature_K": sol.y[self.i_T],
            "sei_thickness_m": sol.y[self.i_L],
            "capacity_loss_Ah": sol.y[self.i_Q] / 3600.0,
            "j_sei_A_m2": j_sei_values,
            "current_A": current_values,
        }

    def run_cycles(
        self,
        n_cycles=1,
        charge_time=3600.0,
        discharge_time=3600.0,
        current=None,               # Kept for backwards compatibility
        points_per_phase=11,
        V_max=None,
        V_min=None,
        pre_discharge=True,
        I_charge=1.5,
        I_discharge=5.0,
        enable_cv=False,
        cv_time=4 * 3600.0,
        I_cv_cutoff=0.04865,
    ):
        if V_max is None:
            V_max = self.sim.upper_cut_off_V
        if V_min is None:
            V_min = self.sim.lower_cut_off_V
        if n_cycles < 1 or points_per_phase < 2:
            raise ValueError("n_cycles >= 1 and points_per_phase >= 2 required")
        if V_min >= V_max or min(charge_time, discharge_time, cv_time) <= 0:
            raise ValueError("Invalid voltage or time bounds")
        if min(I_charge, I_discharge, I_cv_cutoff) <= 0:
            raise ValueError("All current magnitudes must be positive")
        if enable_cv and I_cv_cutoff >= I_charge:
            raise ValueError("CV cutoff current must be lower than CC charging current")

        first_run = self._last_state is None
        y = self.initial_state() if first_run else self._last_state.copy()
        t = 0.0 if first_run else self._last_time
        results = []
        discharge_capacities = []
        original_current = self.sim.I_app

        def save_phase(sol, label, cycle):
            diag = self.diagnostics(sol)
            diag["cycle"] = np.full(sol.t.size, cycle)
            diag["phase"] = np.full(sol.t.size, label)
            results.append(diag)

        try:
            self._control_mode = "CC"
            if pre_discharge and first_run:
                self.sim.I_app = I_discharge
                pre_sol = self._integrate_interval(
                    t_start=t, t_end=t + discharge_time,
                    y0=y, num_points=points_per_phase,
                    voltage_cutoff=V_min, direction=-1,
                )
                y = pre_sol.y[:, -1].copy()
                t = float(pre_sol.t[-1])

            for cycle in range(1, n_cycles + 1):
                # Stage 1: constant-current charge to V_max.
                self._control_mode = "CC"
                self.sim.I_app = -I_charge
                V_start = self._terminal_voltage(y)
                if V_start >= V_max:
                    raise RuntimeError(
                        f"Cycle {cycle}: CC charge starts at {V_start:.4f} V >= {V_max} V"
                    )
                cc = self._integrate_interval(
                    t_start=t, t_end=t + charge_time,
                    y0=y, num_points=points_per_phase,
                    voltage_cutoff=V_max, direction=+1,
                )
                save_phase(cc, "charge_CC", cycle)
                y, t = cc.y[:, -1].copy(), float(cc.t[-1])

                # Stage 2: constant-voltage charge until |I| <= I_cv_cutoff.
                if enable_cv:
                    self._control_mode = "CV"
                    self._cv_voltage = float(V_max)
                    self._cv_max_current = float(I_charge)
                    cv = self._integrate_interval(
                        t_start=t, t_end=t + cv_time,
                        y0=y, num_points=points_per_phase,
                        current_cutoff=I_cv_cutoff,
                    )
                    save_phase(cv, "charge_CV", cycle)
                    y, t = cv.y[:, -1].copy(), float(cv.t[-1])

                # Stage 3: constant-current discharge to V_min.
                self._control_mode = "CC"
                self.sim.I_app = I_discharge
                V_start = self._terminal_voltage(y)
                if V_start <= V_min:
                    raise RuntimeError(
                        f"Cycle {cycle}: discharge starts at {V_start:.4f} V <= {V_min} V"
                    )
                discharge = self._integrate_interval(
                    t_start=t, t_end=t + discharge_time,
                    y0=y, num_points=points_per_phase,
                    voltage_cutoff=V_min, direction=-1,
                )
                save_phase(discharge, "discharge", cycle)
                actual_duration = float(discharge.t[-1] - t)
                discharge_capacities.append(I_discharge * actual_duration / 3600.0)
                y, t = discharge.y[:, -1].copy(), float(discharge.t[-1])

            capacities = np.asarray(discharge_capacities)
            Q_reference = capacities[0]
            output = {
                key: np.concatenate([r[key] for r in results])
                for key in results[0]
            }
            output["cycle_summary"] = {
                "cycle": np.arange(1, n_cycles + 1),
                "discharge_capacity_Ah": capacities,
                "SOH_percent": 100 * capacities / Q_reference,
                "capacity_fade_percent": 100 * (1 - capacities / Q_reference),
            }
            self._last_state = y.copy()
            self._last_time = t
            return output
        finally:
            self._control_mode = "CC"
            self._cv_voltage = None
            self._cv_max_current = None
            self.sim.I_app = original_current
