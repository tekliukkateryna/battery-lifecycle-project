import numpy as np

from backend.physics.config import BatteryGeometry, BatteryParameters, SimulationConfig

# додати логіку для катода, бо тут все під анод написано
class SolidDiffusion:

    def __init__(self, geo: BatteryGeometry, param: BatteryParameters, sim: SimulationConfig, electrode):
        self.geo = geo
        self.param = param
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


    def compute_derivatives(self, c_s, T):

        geo = self.geo
        param = self.param
        sim = self.sim

        dc_s_dt = np.zeros_like(c_s)

        dr = self.r[1]-self.r[0]

        # Арреніус
        Dk = self.D_ref * np.exp(
            -(self.Ea / param.R) * (1.0 / T - 1.0 / 298.15)
        )

        # Внутрішні вузли
        d2c_dr2 = (c_s[2:] - 2 * c_s[1:-1] + c_s[:-2]) / dr**2
        dc_dr = (c_s[2:] - c_s[:-2]) / (2 * dr)
        dc_s_dt[1:-1] = Dk * (d2c_dr2 + (2 / self.r[1:-1]) * dc_dr) ##########

        # Гранична умова в центрі (r = 0)
        dc_s_dt[0] = 3 * Dk * (2 * (c_s[1] - c_s[0])) / dr**2

        # Гранична умова на поверхні (r = Rk)
        iapp_now = sim.Iapp
        jk = self.sign * iapp_now / self.Lk
        surface_flux = jk / (self.b_k * param.F)

        dc_s_dt[-1] = (2 * Dk / dr**2 * (c_s[-2] - c_s[-1] - dr * (surface_flux / Dk)) + (2 * Dk / self.Rk) * (-surface_flux / Dk))

        return dc_s_dt

class ElectrolyteDiffusionSPMe:

    def __init__(self, geo: BatteryGeometry, param: BatteryParameters, sim: SimulationConfig):

        self.geo = geo
        self.param = param
        self.sim = sim

        # Вектор пористостей eps(x)
        self.eps = np.concatenate([
            np.full(geo.N_x_n, geo.eps_n),
            np.full(geo.N_x_s, geo.eps_s),
            np.full(geo.N_x_p, geo.eps_p),
        ])

        # Вектор коефіцієнтів Бруггемана B(x)
        self.B_x = np.concatenate([
            np.full(geo.N_x_n, geo.B_n),
            np.full(geo.N_x_s, geo.B_s),
            np.full(geo.N_x_p, geo.B_p),
        ])

        # Вектор геометричних властивостей матеріалу b(x)
        self.b_x = np.concatenate([
            np.full(geo.N_x_n, param.get_b_n(geo)),
            np.full(geo.N_x_s, 0),
            np.full(geo.N_x_p, param.get_b_p(geo)),
        ])

    def compute_i_e(self, Iapp):
        geo = self.geo

        L_total = geo.L_n + geo.L_s + geo.L_p
        i_e = np.zeros_like(geo.x_e)

        # Анод: (I_app / L_n) * x
        mask_n = geo.x_e <= geo.L_n
        i_e[mask_n] = (Iapp / geo.L_n) * geo.x_e[mask_n]

        # Сепаратор: I_app
        mask_s = (geo.x_e > geo.L_n) & (geo.x_e < (L_total - geo.L_p))
        i_e[mask_s] = Iapp

        # Катод: (I_app / L_p) * (L - x)
        mask_p = geo.x_e >= (L_total - geo.L_p)
        i_e[mask_p] = (Iapp / geo.L_p) * (L_total - geo.x_e[mask_p])

        return i_e

    def compute_j_x(self, Iapp):

        geo = self.geo

        L_total = geo.L_n + geo.L_s + geo.L_p
        j_x = np.zeros_like(geo.x_e)

        # Анод: j_n = I_app / L_n
        mask_n = geo.x_e <= geo.L_n
        j_x[mask_n] = Iapp / geo.L_n

        # Сепаратор: j_s = 0 (залишаються нулі)

        # Катод: j_p = -I_app / L_p
        mask_p = geo.x_e >= (L_total - geo.L_p)
        j_x[mask_p] = -Iapp / geo.L_p

        return j_x

    def compute_derivatives(self, c_e, T):

        geo = self.geo
        param = self.param
        sim = self.sim

        # Автоматичний розрахунок i_e та j_x для поточних параметрів
        i_e_x = self.compute_i_e(sim.Iapp)
        j_x = self.compute_j_x(sim.Iapp)

        # D_e(c_e, T) з урахуванням температури (Арреніус)
        D_e_bulk = param.D_e_ref * np.exp(
            -(param.Ea_De / param.R) * (1 / T - 1 / 298.15)
        )

        # Ефективна дифузія D_eff(x) = D_e * B(x)
        D_eff = D_e_bulk * self.B_x

        # Дифузійний потік (Гармонійне середнє на межах)
        D_interface = 2 * D_eff[:-1] * D_eff[1:] / (D_eff[:-1] + D_eff[1:])
        diff_flux = -D_interface * (c_e[1:] - c_e[:-1]) / geo.dx

        # Міграційний потік (Усереднення i_e на межах)
        i_e_interface = 0.5 * (i_e_x[:-1] + i_e_x[1:])
        migration_flux = (param.t_plus / param.F) * i_e_interface

        # Повний потік J
        total_flux = diff_flux - migration_flux

        # Граничні умови - нульовий потік на стінках x=0 та x=L
        flux_full = np.pad(
            total_flux, (1, 1), mode="constant", constant_values=0
        )

        # Джерельний доданок
        source_term = (self.b_x * j_x) / param.F

        # Остаточне рівняння: dc_e/dt
        dc_e_dt = (
            -(flux_full[1:] - flux_full[:-1]) / (geo.dx * self.eps)
        ) + (source_term / self.eps)

        return dc_e_dt

#Треба дописать OCV криві бо це функція або таблиця, я там це маю в одній статті про це прочитати
def OCV_n(v):
    return v

def OCV_p(v):
    return v

class TerminalVoltage:
    def __init__(self, geo: BatteryGeometry, param: BatteryParameters, sim: SimulationConfig, ocv_model):
        self.param = param
        self.geo = geo
        self.sim = sim
        self.ocv = ocv_model

    def Ueq(self, c_s_n, c_s_p):

        param=self.param

        c_surf_n_avg = c_s_n[-1]
        c_surf_p_avg = c_s_p[-1]

        theta_n = c_surf_n_avg / param.c_n_max
        theta_p = c_surf_p_avg / param.c_p_max

        U_n = self.ocv.OCV_n(theta_n)
        U_p = self.ocv.OCV_p(theta_p)

        return U_p - U_n

    def compute_j0_x(self, c_e_slice, c_s_surf, K_k, c_k_max) -> np.ndarray:

        param = self.param

        # Безрозмірні співвідношення (стохіометрії)
        c_e_ratio = c_e_slice / param.c_e_0  # 1D-масив розмірністю (N_x_k,)
        stoich = c_s_surf / c_k_max  # Скаляр (0..1)
        vacancies = 1.0 - stoich  # Скаляр (0..1)
        
        # Вирази під коренем (з захистом від мікроскопічних від'ємних значень через чисельні погрешності)
        radicand = np.maximum(c_e_ratio * stoich * vacancies, 0.0)
        
        # Broadcasting
        return param.F * K_k * np.sqrt(radicand)


    def eta_r(self, c_e, c_s_n, c_s_p, T):

        geo = self.geo
        param = self.param
        sim = self.sim

        j_n = sim.Iapp / geo.L_n
        j_p = -sim.Iapp / geo.L_p

        c_e_n = c_e[: geo.N_x_n]
        c_e_p = c_e[geo.N_x_n + geo.N_x_s :]
        
        # Беремо поверхневу концентрацію з SolidDiffusion (останній елемент радіальної сітки r=R_k)
        c_sn_surf = c_s_n[-1]
        c_sp_surf = c_s_p[-1]
        
        # Обчислюємо 1D-вектори j_n0(x, t) та j_p0(x, t)[cite: 1]
        j_n0_x = self.compute_j0_x(
            c_e_slice = c_e_n,
            c_s_surf = c_sn_surf,
            K_k = param.K_n,
            c_k_max = param.c_n_max,
            )
        
        j_p0_x = self.compute_j0_x(
            c_e_slice = c_e_p,
            c_s_surf = c_sp_surf,
            K_k = param.K_p,
            c_k_max = param.c_p_max,
            )
        
        # Усереднюємо для підстановки у формулу напруги
        j_n0_avg = np.mean(j_n0_x)
        j_p0_avg = np.mean(j_p0_x)

        res = (2 * param.R * T) / param.F * (np.arcsinh(j_n / j_n0_avg) - np.arcsinh(j_p / j_p0_avg))
        return res


    #def calculate(self):

        ##################### Це я зараз пишу #################

        # Напруга батареї
        #V = U_eq - eta_r - eta_c - delta_phi_e - delta_phi_s

        #return V