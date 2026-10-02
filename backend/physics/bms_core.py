from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
import math


class Measurement:
    def __init__(self, timestamp: float, valid: bool):
        self._timestamp = timestamp
        self._valid = valid

    def is_expired(self, current_time: float, max_delay: float) -> bool:
        if current_time < self._timestamp:
            return True
        return (current_time - self._timestamp) > max_delay

    def invalidate_if_stale(self, current_time: float, max_age: float) -> bool:
        if self.is_expired(current_time, max_age):
            self._valid = False
            return True
        return False

    def is_valid(self) -> bool:
        return self._valid

    def get_timestamp(self) -> float:
        return self._timestamp


class ThermalReading(Measurement):
    def __init__(self, timestamp: float, valid: bool, temp_c: float, ambient_c: float):
        super().__init__(timestamp, valid)
        self._temp_c = temp_c
        self._ambient_c = ambient_c

    def get_gradient(self) -> float:
        return abs(self._temp_c - self._ambient_c)

    def to_kelvin(self) -> float:
        if self._temp_c < -273.15:
            return 0.0
        return self._temp_c + 273.15

    def get_temp(self) -> float:
        return self._temp_c


class VoltageReading(Measurement):
    def __init__(self, timestamp: float, valid: bool, cell_v: float, bus_v: float):
        super().__init__(timestamp, valid)
        self._cell_v = cell_v
        self._bus_v = bus_v

    def compute_deviation_ratio(self, total_cells: int) -> float:
        if total_cells <= 0 or self._bus_v <= 0.0:
            return 0.0
        avg_v = self._bus_v / total_cells
        return (self._cell_v - avg_v) / avg_v

    def is_within_bounds(self, v_min: float, v_max: float) -> bool:
        return v_min <= self._cell_v <= v_max and self.is_valid()

    def get_cell_v(self) -> float:
        return self._cell_v


class HealthReport:
    def __init__(self, soh_percent: float, estimated_r_int: float, cycle_count: int):
        self._soh_percent = soh_percent
        self._r_int = estimated_r_int
        self._cycles = cycle_count

    def is_end_of_life(self, r_threshold: float) -> bool:
        return self._soh_percent <= 80.0 or self._r_int >= r_threshold

    def estimate_remaining_cycles(self, degradation_rate: float) -> int:
        if degradation_rate <= 0.0:
            return 0
        usable_soh = self._soh_percent - 80.0
        if usable_soh <= 0.0:
            return 0
        return int(usable_soh / degradation_rate)


# ---------------- SENSORS HIERARCHY ----------------

class AbstractSensor(ABC):
    def __init__(self, sensor_id: str, error_bias: float):
        self._sensor_id = sensor_id
        self._bias = error_bias

    @abstractmethod
    def sample_raw(self) -> float:
        pass

    @abstractmethod
    def run_self_test(self) -> bool:
        pass

    def calibrate(self, ref_val: float, raw_val: float) -> None:
        self._bias = raw_val - ref_val

    def get_calibrated_value(self) -> float:
        return self.sample_raw() - self._bias


class VoltageSensor(AbstractSensor):
    def __init__(self, sensor_id: str, bias: float, max_v: float):
        super().__init__(sensor_id, bias)
        self._max_v = max_v

    def sample_raw(self) -> float:
        return 3.65 + self._bias + 0.005

    def run_self_test(self) -> bool:
        val = self.sample_raw()
        return 0.0 <= val <= self._max_v

    def is_saturated(self) -> bool:
        return self.sample_raw() >= (self._max_v * 0.98)


class TemperatureSensor(AbstractSensor):
    def __init__(self, sensor_id: str, bias: float, warn_t: float):
        super().__init__(sensor_id, bias)
        self._warn_t = warn_t

    def sample_raw(self) -> float:
        return 34.5 + self._bias

    def run_self_test(self) -> bool:
        t = self.get_calibrated_value()
        return -40.0 < t < 125.0

    def is_overheating_imminent(self) -> bool:
        return self.get_calibrated_value() >= (self._warn_t - 3.0)


class CurrentSensor(AbstractSensor):
    def __init__(self, sensor_id: str, bias: float, r_shunt: float):
        super().__init__(sensor_id, bias)
        self._r_shunt = r_shunt

    def sample_raw(self) -> float:
        if self._r_shunt <= 0.0:
            return 0.0
        v_drop = 0.025
        return (v_drop / self._r_shunt) + self._bias

    def run_self_test(self) -> bool:
        return 0.0001 < self._r_shunt < 0.1

    def compute_shunt_dissipation(self) -> float:
        current = self.get_calibrated_value()
        return (current ** 2) * self._r_shunt


# ---------------- ESTIMATORS HIERARCHY ----------------

class AbstractStateEstimator(ABC):
    def __init__(self, nominal_cap: float, initial_soc: float):
        self._nominal_cap = nominal_cap
        self._soc = initial_soc

    @abstractmethod
    def update(self, current: float, voltage: float, dt: float) -> None:
        pass

    @abstractmethod
    def reset_by_ocv(self, ocv: float) -> None:
        pass

    def get_residual_ah(self) -> float:
        return self._nominal_cap * (self._soc / 100.0)

    def clamp_soc(self) -> None:
        self._soc = max(0.0, min(100.0, self._soc))

    def get_soc(self) -> float:
        return self._soc


class AhCountingEstimator(AbstractStateEstimator):
    def __init__(self, cap: float, soc: float, eta: float):
        super().__init__(cap, soc)
        self._coulomb_eff = eta

    def update(self, current: float, voltage: float, dt: float) -> None:
        eff = self._coulomb_eff if current < 0.0 else 1.0
        delta_ah = (current * dt / 3600.0) * eff
        self._soc -= (delta_ah / self._nominal_cap) * 100.0
        self.clamp_soc()

    def reset_by_ocv(self, ocv: float) -> None:
        if ocv >= 4.15:
            self._soc = 100.0
        elif ocv <= 3.0:
            self._soc = 0.0
        else:
            self._soc = (ocv - 3.0) / (4.15 - 3.0) * 100.0

    def compute_loss(self, current: float, dt: float) -> float:
        if current >= 0.0:
            return 0.0
        total_in = -current * dt / 3600.0
        return total_in * (1.0 - self._coulomb_eff)


class AdaptiveFilterEstimator(AbstractStateEstimator):
    def __init__(self, cap: float, soc: float, q: float, r: float, p: float):
        super().__init__(cap, soc)
        self._q = q
        self._r = r
        self._p = p

    def update(self, current: float, voltage: float, dt: float) -> None:
        delta_soc = -(current * dt / (self._nominal_cap * 3600.0)) * 100.0
        self._soc += delta_soc
        self._p += self._q

        v_est = 3.0 + 0.012 * self._soc
        innovation = voltage - v_est

        h = 0.012
        kalman_gain = (self._p * h) / (h * self._p * h + self._r)
        self._soc += kalman_gain * innovation
        self._p = (1.0 - kalman_gain * h) * self._p
        self.clamp_soc()

    def reset_by_ocv(self, ocv: float) -> None:
        raw_soc = (ocv - 3.0) / 0.012
        self._soc = max(0.0, min(100.0, raw_soc))
        self._p = 1.0

    def is_converged(self, threshold: float) -> bool:
        return self._p < threshold


# ---------------- FAULT DETECTORS HIERARCHY ----------------

class AbstractFaultDetector(ABC):
    def __init__(self, max_faults: int):
        self._fault_count = 0
        self._max_faults = max_faults

    @abstractmethod
    def check_condition(self, v: float, t: float, dt: float) -> bool:
        pass

    def mitigate_step(self) -> None:
        if self._fault_count > 0:
            self._fault_count -= 1

    def get_severity(self) -> int:
        if self._fault_count == 0:
            return 0
        if self._fault_count < self._max_faults:
            return 1
        return 2

    def is_tripped(self) -> bool:
        return self._fault_count >= self._max_faults


class OvervoltageDetector(AbstractFaultDetector):
    def __init__(self, max_faults: int, crit_v: float):
        super().__init__(max_faults)
        self._crit_v = crit_v

    def check_condition(self, v: float, t: float, dt: float) -> bool:
        if v > self._crit_v:
            self._fault_count += 1
            return True
        self.mitigate_step()
        return False

    def get_headroom(self, current_v: float) -> float:
        return self._crit_v - current_v


class ThermalRunawayDetector(AbstractFaultDetector):
    def __init__(self, max_faults: int, max_rate: float):
        super().__init__(max_faults)
        self._max_rate = max_rate
        self._last_t = 25.0

    def check_condition(self, v: float, t: float, dt: float) -> bool:
        if dt <= 0.0:
            return False
        rate = (t - self._last_t) / dt
        self._last_t = t

        if rate > self._max_rate:
            self._fault_count += 2
            return True
        self.mitigate_step()
        return False

    def extrapolate_temp(self, dt: float) -> float:
        return self._last_t + (self._max_rate * dt)


# ---------------- COOLING CONTROLLERS HIERARCHY ----------------

class AbstractCoolingController(ABC):
    def __init__(self):
        self._duty = 0.0
        self._active = False

    @abstractmethod
    def regulate(self, temp: float, dt: float) -> None:
        pass

    def emergency_stop(self) -> None:
        self._duty = 0.0
        self._active = False

    def get_power_consumption(self, max_w: float) -> float:
        if not self._active:
            return 0.0
        return max_w * (self._duty / 100.0)

    def get_duty(self) -> float:
        return self._duty


class BinaryFanController(AbstractCoolingController):
    def __init__(self, target_t: float):
        super().__init__()
        self._target_t = target_t

    def regulate(self, temp: float, dt: float) -> None:
        hysteresis = 2.0
        if temp > self._target_t + hysteresis:
            self._duty = 100.0
            self._active = True
        elif temp < self._target_t - hysteresis:
            self._duty = 0.0
            self._active = False

    def adjust_target(self, new_target: float) -> bool:
        if 10.0 <= new_target <= 60.0:
            self._target_t = new_target
            return True
        return False


class PwmPumpController(AbstractCoolingController):
    def __init__(self, kp: float, ki: float):
        super().__init__()
        self._kp = kp
        self._ki = ki
        self._integrator = 0.0

    def regulate(self, temp: float, dt: float) -> None:
        target = 30.0
        error = temp - target
        if error <= 0.0:
            self._duty = 0.0
            self._active = False
            self._integrator = 0.0
            return

        self._integrator += error * dt
        self._integrator = max(-50.0, min(50.0, self._integrator))

        out = (self._kp * error) + (self._ki * self._integrator)
        self._duty = max(10.0, min(100.0, out))
        self._active = True

    def flush_integrator(self) -> None:
        self._integrator = 0.0


# ---------------- GENERICS & DATA STRUCTURES ----------------

T = TypeVar('T', int, float)

class CircularBuffer(Generic[T]):
    def __init__(self, capacity: int):
        self._cap = capacity
        self._buf: List[Optional[T]] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._count = 0

    def push(self, item: T) -> None:
        self._buf[self._head] = item
        self._head = (self._head + 1) % self._cap
        if self._count < self._cap:
            self._count += 1
        else:
            self._tail = (self._tail + 1) % self._cap

    def pop(self) -> Optional[T]:
        if self._count == 0:
            return None
        val = self._buf[self._tail]
        self._buf[self._tail] = None
        self._tail = (self._tail + 1) % self._cap
        self._count -= 1
        return val

    def at_relative(self, index: int) -> Optional[T]:
        if index < 0 or index >= self._count:
            return None
        return self._buf[(self._tail + index) % self._cap]

    def clear(self) -> None:
        self._buf = [None] * self._cap
        self._head = 0
        self._tail = 0
        self._count = 0

    def size(self) -> int:
        return self._count


class MetricAggregator(Generic[T]):
    def __init__(self):
        self._data: List[T] = []

    def append(self, val: T) -> None:
        v = float(val)
        if not math.isnan(v) and not math.isinf(v):
            self._data.append(val)

    def calculate_mean(self) -> float:
        if not self._data:
            return 0.0
        return sum(self._data) / len(self._data)

    def calculate_std_dev(self) -> float:
        if len(self._data) < 2:
            return 0.0
        mean = self.calculate_mean()
        var = sum((x - mean) ** 2 for x in self._data) / (len(self._data) - 1)
        return math.sqrt(var)

    def compute_range(self) -> float:
        if len(self._data) < 2:
            return 0.0
        return float(max(self._data) - min(self._data))


# ---------------- POLYMORPHIC CLIENT METHODS ----------------

def poll_sensors(sensors: List[AbstractSensor]) -> List[float]:
    results = []
    for s in sensors:
        if s.run_self_test():
            results.append(s.get_calibrated_value())
        else:
            results.append(-999.0)
    return results

def compare_estimators(est1: AbstractStateEstimator, est2: AbstractStateEstimator,
                       current: float, voltage: float, dt: float) -> tuple[float, float]:
    est1.update(current, voltage, dt)
    est2.update(current, voltage, dt)
    return est1.get_soc(), est2.get_soc()

def scan_faults(detectors: List[AbstractFaultDetector], v: float, t: float, dt: float) -> bool:
    trip_flag = False
    for d in detectors:
        if d.check_condition(v, t, dt) and d.is_tripped():
            trip_flag = True
    return trip_flag

def process_cooling(controller: AbstractCoolingController, temp: float, dt: float) -> float:
    controller.regulate(temp, dt)
    return controller.get_duty()

def run_buffer_pipeline(buf: CircularBuffer[T], agg: MetricAggregator[T], stream: List[T]) -> None:
    for item in stream:
        buf.push(item)
    while buf.size() > 0:
        val = buf.pop()
        if val is not None:
            agg.append(val)