import numpy as np

class SolidDiffusion:

    def __init__(self, N_r, Rk, Dk, Iapp, Lk, epsilon):
        self.N_r = N_r
        self.Rk = Rk
        self.Dk = Dk
        self.Iapp = Iapp
        self.Lk = Lk
        self.Fc = 96485.33
        self.b_k = 1.0 - epsilon

        self.r = np.linspace(0, Rk, N_r)
        self.dr = self.r[1] - self.r[0]

    def compute_derivatives(self, t, Ck, T):

        R = 8.31446
        dc_dt = np.zeros_like(Ck)

        # Арреніус
        Dk = self.D_s_ref * np.exp(
            -(self.Ea_Ds / R) * (1.0 / T - 1.0 / 298.15)
        )

        # Внутрішні вузли
        d2c_dr2 = (Ck[2:] - 2 * Ck[1:-1] + Ck[:-2]) / self.dr**2
        dc_dr = (Ck[2:] - Ck[:-2]) / (2 * self.dr)
        dc_dt[1:-1] = Dk * (d2c_dr2 + (2 / self.r[1:-1]) * dc_dr)

        # Гранична умова в центрі (r = 0)
        dc_dt[0] = 3 * Dk * (2 * (Ck[1] - Ck[0])) / self.dr**2

        # Гранична умова на поверхні (r = Rk)
        iapp_now = self.Iapp(t) if callable(self.Iapp) else self.Iapp
        jk = iapp_now / self.Lk
        surface_flux = jk / (self.b_k * self.Fc)

        dc_dt[-1] = (2 * Dk / self.dr**2 * (Ck[-2] - Ck[-1] - self.dr * (surface_flux / Dk)) + (2 * Dk / self.Rk) * (-surface_flux / Dk))

        return dc_dt

class ElectrolyteDiffusionSPMe:

    def __init__(self, geometry, parameters, config):
        self.N_x_n = geometry.N_x_n
        self.N_x_s = geometry.N_x_s
        self.N_x_p = geometry.N_x_p
        self.dx = geometry.dx

        self.F = parameters.F
        self.t_plus = parameters.t_plus

        # Вектор пористостей eps(x)
        self.eps = np.concatenate([
            np.full(self.N_x_n, geometry.eps_n),
            np.full(self.N_x_s, geometry.eps_s),
            np.full(self.N_x_p, geometry.eps_p),
        ])

        # Вектор коефіцієнтів Бруггемана B(x)
        self.B_x = np.concatenate([
            np.full(self.N_x_n, parameters.B_n),
            np.full(self.N_x_s, parameters.B_s),
            np.full(self.N_x_p, parameters.B_p),
        ])

        # Вектор геометричних властивостей матеріалу b(x)
        self.b_x = np.concatenate([
            np.full(self.N_x_n, parameters.b_n),
            np.full(self.N_x_s, parameters.b_s),
            np.full(self.N_x_p, parameters.b_p),
        ])

        self.Iapp = config.Iapp

        self.L_n = geometry.L_n
        self.L_s = geometry.L_s
        self.L_p = geometry.L_p

        # Створення сітки координат x_grid (у центрах комірок)
        N_total = self.N_x_n + self.N_x_s + self.N_x_p
        self.x_grid = (np.arange(N_total) + 0.5) * self.dx

        # Температурні параметри для D_e(T)
        self.D_e_ref = parameters.D_e_ref
        self.Ea_De = parameters.Ea_De

    @staticmethod
    def compute_i_e(x_grid, I_app, L_n, L_s, L_p):
        L_total = L_n + L_s + L_p
        i_e = np.zeros_like(x_grid)

        # Анод: (I_app / L_n) * x
        mask_n = x_grid <= L_n
        i_e[mask_n] = (I_app / L_n) * x_grid[mask_n]

        # Сепаратор: I_app
        mask_s = (x_grid > L_n) & (x_grid < (L_total - L_p))
        i_e[mask_s] = I_app

        # Катод: (I_app / L_p) * (L - x)
        mask_p = x_grid >= (L_total - L_p)
        i_e[mask_p] = (I_app / L_p) * (L_total - x_grid[mask_p])

        return i_e

    @staticmethod
    def compute_j_x(x_grid, I_app, L_n, L_s, L_p):
        L_total = L_n + L_s + L_p
        j_x = np.zeros_like(x_grid)

        # Анод: j_n = I_app / L_n
        mask_n = x_grid <= L_n
        j_x[mask_n] = I_app / L_n

        # Сепаратор: j_s = 0 (залишаються нулі)

        # Катод: j_p = -I_app / L_p
        mask_p = x_grid >= (L_total - L_p)
        j_x[mask_p] = -I_app / L_p

        return j_x

    def compute_derivatives(self, Ck, T):
        R = 8.314

        # Автоматичний розрахунок i_e та j_x для поточних параметрів
        i_e_x = self.compute_i_e(
            self.x_grid, self.Iapp, self.L_n, self.L_s, self.L_p
        )
        j_x = self.compute_j_x(
            self.x_grid, self.Iapp, self.L_n, self.L_s, self.L_p
        )

        # D_e(c_e, T) з урахуванням температури (Арреніус)
        D_e_bulk = self.D_e_ref * np.exp(
            -(self.Ea_De / R) * (1 / T - 1 / 298.15)
        )

        # Ефективна дифузія D_eff(x) = D_e * B(x)
        D_eff = D_e_bulk * self.B_x

        # Дифузійний потік (Гармонійне середнє на межах)
        D_interface = 2 * D_eff[:-1] * D_eff[1:] / (D_eff[:-1] + D_eff[1:])
        diff_flux = -D_interface * (Ck[1:] - Ck[:-1]) / self.dx

        # Міграційний потік (Усереднення i_e на межах)
        i_e_interface = 0.5 * (i_e_x[:-1] + i_e_x[1:])
        migration_flux = (self.t_plus / self.F) * i_e_interface

        # Повний потік J
        total_flux = diff_flux - migration_flux

        # Граничні умови - нульовий потік на стінках x=0 та x=L
        flux_full = np.pad(
            total_flux, (1, 1), mode="constant", constant_values=0
        )

        # Джерельний доданок
        source_term = (self.b_x * j_x) / self.F

        # Остаточне рівняння: dc_e/dt
        dc_e_dt = (
            -(flux_full[1:] - flux_full[:-1]) / (self.dx * self.eps)
        ) + (source_term / self.eps)

        return dc_e_dt


#Треба дописать OCV криві бо це функція або таблиця, я там це маю в одній статті про це прочитати
class TerminalVoltage:
    def __init__(self, parameters):
        self.R = parameters.R
        self.F = parameters.F
        self.T = parameters.T

    #def calculate(self):

        ##################### Це я зараз пишу #################

        # Рівноважна напруга (OCV)
        

        # Кінетична перенапруга
        

        # Концентраційна перенапруга (просто середнє значення c_e по геометрії)
        

        # Омічні втрати (константний опір * струм)
        

        # Напруга батареї
        #V = U_eq - eta_r - eta_c - delta_phi_e - delta_phi_s

        #return V

#class ThermalBalance:
    


#class SeiDegradation: