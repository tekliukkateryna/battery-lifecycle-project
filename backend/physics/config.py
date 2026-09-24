from dataclasses import dataclass
import numpy as np


@dataclass
class BatteryGeometry:

    # Кількість точок дискретизації сферичних частинок по радіусу (r)
    N_r_n: int = 10  # Частинка анода
    N_r_p: int = 10  # Частинка катода

    # Товщини зон [м]
    L_n: float = 80e-6  # Анод
    L_s: float = 25e-6  # Сепаратор
    L_p: float = 80e-6  # Катод

    # Радіуси частинок [м]
    R_n: float = 10e-6  # Анод
    R_p: float = 10e-6  # Катод

    # Кількість точок просторової сітки по товщині (x)
    dx: float = 10e-6

    # Пористості (об'ємна частка рідкого електроліту)
    eps_n: float = 0.3
    eps_s: float = 0.4
    eps_p: float = 0.3

    @property
    def N_x_n(self) -> int:
        return int(np.round(self.L_n / self.dx))

    @property
    def N_x_s(self) -> int:
        return int(np.round(self.L_s / self.dx))

    @property
    def N_x_p(self) -> int:
        return int(np.round(self.L_p / self.dx))

    @property
    def N_x_total(self) -> int:
        return self.N_x_n + self.N_x_s + self.N_x_p


@dataclass
class BatteryParameters:
    # Константа Фарадея та газова константа
    F: float = 96485.33
    R: float = 8.31446

    # Референсні коефіцієнти дифузії при T = 298.15 K [м²/с]
    D_s_n_ref: float = 3.9e-14  # Тверда фаза анода
    D_s_p_ref: float = 1.0e-14  # Тверда фаза катода
    D_e_ref: float = 2.6e-10  # Електроліт

    # Енергії активації (Арреніус) [Дж/моль]
    Ea_Ds_n: float = 35000.0
    Ea_Ds_p: float = 29000.0
    Ea_De: float = 15000.0
    Ea_sei: float = 38000.0

    # Коефіцієнти Бруггемана
    B_n: float = 0.3**1.5  # B_n
    B_s: float = 0.4**1.5  # B_s
    B_p: float = 0.3**1.5  # B_p

    # Специфічні геометрії $b(x)$ для SPMe
    # b_k = 3 * (1 - eps_k) / R_p_k
    b_n: float = 3 * (1.0 - 0.3) / 10e-6  # Анод
    b_s: float = 0.0  # У сепараторі немає твердої фази
    b_p: float = 3 * (1.0 - 0.3) / 10e-6  # Катод

    # Параметри деградації SEI
    k_sei_ref: float = 1e-12  # Константа реакції [м/с]
    M_sei: float = 0.162  # Молярна маса [кг/моль]
    rho_sei: float = 2100.0  # Густина [кг/м³]

    # Електрохімія
    t_plus: float = 0.38  # Число переносу іонів Li+

    # Теплові параметри
    mass: float = 0.04  # Маса [кг]
    Cp: float = 1100.0  # Теплоємність [Дж/(кг·К)]
    h_cool: float = 12.0  # Конвекція [Вт/(м²·К)]
    A_surface: float = 0.05  # Площа охолодження [м²]
    T_amb: float = 298.15  # Довкілля [К]


class SimulationConfig:

    def __init__(
        self, geometry: BatteryGeometry, parameters: BatteryParameters
    ):
        self.Iapp = 2.5  # Прикладений струм [А]
        self.t_final = 3600.0  # Час симуляції [с]

        # Початкові концентрації [моль/м³]
        self.Ck_n_init = 25000.0  # Літій в аноді
        self.Ck_p_init = 10000.0  # Літій в катоді
        self.Ce_init = 1000.0  # Електроліт

        self.delta_sei_init = 5e-9  # Товщина SEI [м]
        self.T_init = parameters.T_amb  # Початкова температура [К]