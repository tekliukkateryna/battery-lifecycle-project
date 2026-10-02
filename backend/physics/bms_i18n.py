import json
import time
from typing import Dict, Any


class LocalizationManager:
    _CATALOG: Dict[str, Dict[str, str]] = {
        "uk": {
            "init": "Запуск комплексу моніторингу BMS...",
            "status_ok": "Система в нормі, аварійних спрацьовувань немає.",
            "status_trip": "УВАГА: Спрацював захист! Виявлено аварійний стан.",
            "sensor_v": "Напруга комірки: {val:.3f} В",
            "cooling_info": "Актуатор охолодження: {duty:.1f}%",
            "stat_summary": "Статистика шини: середнє = {mean:.3f} В, СКО = {std:.4f} В"
        },
        "en": {
            "init": "Starting BMS monitoring runtime...",
            "status_ok": "System nominal, no active faults.",
            "status_trip": "WARNING: Trip condition reached! Emergency action needed.",
            "sensor_v": "Cell voltage: {val:.3f} V",
            "cooling_info": "Cooling actuator duty: {duty:.1f}%",
            "stat_summary": "Bus statistics: mean = {mean:.3f} V, std = {std:.4f} V"
        }
    }

    def __init__(self, lang: str = "uk"):
        self._lang = lang if lang in self._CATALOG else "en"

    def switch_language(self, lang: str) -> bool:
        if lang in self._CATALOG:
            self._lang = lang
            return True
        return False

    def msg(self, key: str, **kwargs: Any) -> str:
        text = self._CATALOG.get(self._lang, {}).get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    @property
    def current_lang(self) -> str:
        return self._lang


class DiagnosticLogger:
    def __init__(self, i18n: LocalizationManager):
        self._i18n = i18n
        self._logs: list[dict] = []

    def record_step(self, mean_v: float, soc: float, trip: bool) -> None:
        self._logs.append({
            "timestamp": time.time(),
            "mean_voltage": round(mean_v, 3),
            "soc": round(soc, 2),
            "tripped": trip,
            "message": self._i18n.msg("status_trip" if trip else "status_ok")
        })

    def dump_json(self) -> str:
        return json.dumps({"session_logs": self._logs}, indent=2, ensure_ascii=False)