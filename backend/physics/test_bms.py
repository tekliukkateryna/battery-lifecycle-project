import unittest
from bms_core import (
    Measurement, ThermalReading, VoltageReading, HealthReport,
    VoltageSensor, TemperatureSensor, CurrentSensor,
    AhCountingEstimator, AdaptiveFilterEstimator,
    OvervoltageDetector, ThermalRunawayDetector,
    BinaryFanController, PwmPumpController,
    CircularBuffer, MetricAggregator,
    poll_sensors, compare_estimators, scan_faults, process_cooling, run_buffer_pipeline
)
from bms_i18n import LocalizationManager, DiagnosticLogger


class TestBmsCore(unittest.TestCase):
    def test_measurement_freshness(self):
        m = Measurement(timestamp=100.0, valid=True)
        self.assertFalse(m.is_expired(current_time=104.0, max_delay=5.0))
        self.assertTrue(m.is_expired(current_time=106.0, max_delay=5.0))
        self.assertTrue(m.invalidate_if_stale(current_time=107.0, max_age=5.0))
        self.assertFalse(m.is_valid())

    def test_thermal_values(self):
        t = ThermalReading(0.0, True, 36.6, 20.0)
        self.assertAlmostEqual(t.get_gradient(), 16.6)
        self.assertAlmostEqual(t.to_kelvin(), 309.75)

    def test_health_evaluation(self):
        rep = HealthReport(soh_percent=78.0, estimated_r_int=0.08, cycle_count=600)
        self.assertTrue(rep.is_end_of_life(r_threshold=0.07))
        self.assertEqual(rep.estimate_remaining_cycles(0.05), 0)

    def test_polymorphic_sensors(self):
        sensors = [
            VoltageSensor("V1", 0.0, 4.5),
            TemperatureSensor("T1", 0.0, 60.0),
            CurrentSensor("C1", 0.0, 0.01)
        ]
        vals = poll_sensors(sensors)
        self.assertEqual(len(vals), 3)
        self.assertGreater(vals[0], 0.0)

    def test_ah_soc_integration(self):
        est = AhCountingEstimator(cap=10.0, soc=60.0, eta=1.0)
        est.update(current=10.0, voltage=3.6, dt=360.0) # 1 Ah spent
        self.assertAlmostEqual(est.get_soc(), 50.0)

    def test_kalman_soc_stability(self):
        kf = AdaptiveFilterEstimator(cap=20.0, soc=70.0, q=0.0001, r=0.01, p=0.12)
        for _ in range(50):
            kf.update(current=0.0, voltage=3.84, dt=1.0)
        self.assertTrue(kf.is_converged(0.6))

    def test_detectors_trip(self):
        ov = OvervoltageDetector(max_faults=2, crit_v=4.2)
        ov.check_condition(4.25, 25.0, 1.0)
        self.assertFalse(ov.is_tripped())
        ov.check_condition(4.26, 25.0, 1.0)
        self.assertTrue(ov.is_tripped())

    def test_fan_hysteresis_logic(self):
        fan = BinaryFanController(target_t=40.0)
        fan.regulate(43.0, 1.0)
        self.assertEqual(fan.get_duty(), 100.0)
        fan.regulate(39.0, 1.0)
        self.assertEqual(fan.get_duty(), 100.0)
        fan.regulate(37.0, 1.0)
        self.assertEqual(fan.get_duty(), 0.0)

    def test_circular_buffer_bounds(self):
        buf: CircularBuffer[float] = CircularBuffer(3)
        buf.push(10.0)
        buf.push(20.0)
        buf.push(30.0)
        buf.push(40.0)
        self.assertEqual(buf.size(), 3)
        self.assertEqual(buf.pop(), 20.0)

    def test_i18n_and_logger(self):
        loc = LocalizationManager("uk")
        self.assertIn("Запуск", loc.msg("init"))
        loc.switch_language("en")
        self.assertIn("Starting", loc.msg("init"))

        logger = DiagnosticLogger(loc)
        logger.record_step(3.7, 75.0, False)
        out = logger.dump_json()
        self.assertIn("session_logs", out)


if __name__ == "__main__":
    unittest.main()
