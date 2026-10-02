from bms_core import (
    VoltageSensor, TemperatureSensor, CurrentSensor,
    AhCountingEstimator, AdaptiveFilterEstimator,
    OvervoltageDetector, ThermalRunawayDetector,
    BinaryFanController, PwmPumpController,
    CircularBuffer, MetricAggregator,
    poll_sensors, compare_estimators, scan_faults,
    process_cooling, run_buffer_pipeline
)
from bms_i18n import LocalizationManager, DiagnosticLogger


def main():
    i18n = LocalizationManager("uk")
    logger = DiagnosticLogger(i18n)

    print(f"[{i18n.current_lang.upper()}] {i18n.msg('init')}\n")

    # 1. Поліморфний збір показів сенсорів
    sensors = [
        VoltageSensor("V_CELL_1", 0.005, 4.35),
        TemperatureSensor("T_MOD_1", -0.1, 55.0),
        CurrentSensor("I_SHUNT_1", 0.0, 0.005)
    ]
    raw_vals = poll_sensors(sensors)
    print(f"Дані телеметрії: U={raw_vals[0]:.3f}V, T={raw_vals[1]:.1f}C, I={raw_vals[2]:.2f}A")

    # 2. Оцінювачі заряду
    ah_est = AhCountingEstimator(cap=40.0, soc=80.0, eta=0.98)
    kf_est = AdaptiveFilterEstimator(cap=40.0, soc=80.0, q=0.001, r=0.05, p=0.2)
    s1, s2 = compare_estimators(ah_est, kf_est, current=20.0, voltage=3.65, dt=5.0)
    print(f"Оцінка заряду: Ah-метод={s1:.2f}%, Фільтр Калмана={s2:.2f}%")

    # 3. Перевірка на критичні аварії
    detectors = [
        OvervoltageDetector(max_faults=2, crit_v=4.22),
        ThermalRunawayDetector(max_faults=2, max_rate=2.0)
    ]
    is_fault = scan_faults(detectors, v=4.25, t=35.0, dt=1.0)

    # 4. Охолодження
    pump = PwmPumpController(kp=2.5, ki=0.1)
    duty = process_cooling(pump, temp=36.5, dt=1.0)
    print(i18n.msg("cooling_info", duty=duty))

    # 5. Generic буферизація та статистика
    stream = [3.64, 3.65, 3.66, 3.64, 3.65, 3.67, 3.63]
    buf: CircularBuffer[float] = CircularBuffer(5)
    agg: MetricAggregator[float] = MetricAggregator()
    run_buffer_pipeline(buf, agg, stream)

    mean_v = agg.calculate_mean()
    std_v = agg.calculate_std_dev()
    print(i18n.msg("stat_summary", mean=mean_v, std=std_v))

    # 6. Запис у лог та експорт
    logger.record_step(mean_v, s2, is_fault)
    print("\n--- Сформований JSON звіт ---")
    print(logger.dump_json())


if __name__ == "__main__":
    main()