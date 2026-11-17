"""Validated browser-to-physics request schemas."""
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class LifecycleRequest(BaseModel):
    n_cycles: int = Field(default=100, ge=1, le=10000)
    current: float = Field(default=5.0, gt=0, le=30, description="Cell current magnitude, A")
    temperature_C: float = Field(default=25.0, ge=-30, le=80)
    charge_current_A: float | None = Field(default=None, gt=0, le=30)
    enable_thermal: bool = True
    max_charge_hours: float = Field(default=8.0, gt=0, le=48)
    max_discharge_hours: float = Field(default=8.0, gt=0, le=48)
    initial_soc_percent: float = Field(default=100.0, ge=0, le=100)
    upper_cutoff_V: float = Field(default=4.2, ge=2.0, le=5.0)
    lower_cutoff_V: float = Field(default=2.5, ge=1.5, le=4.5)
    protocol_id: Literal["custom", "imperial_expt5"] = "custom"
    cycling_mode: Literal["CC", "CCCV"] = "CC"
    cv_current_cutoff_A: float = Field(default=0.05, gt=0, le=30)
    max_cv_hours: float = Field(default=4, gt=0, le=48)
    apply_ml: bool = True

    @model_validator(mode="after")
    def validate_cutoffs(self):
        if self.lower_cutoff_V >= self.upper_cutoff_V:
            raise ValueError("lower_cutoff_V must be below upper_cutoff_V")
        return self


class StepRequest(BaseModel):
    mode: Literal["idle", "charge", "discharge"]
    current: float = Field(ge=0, le=30)
    soc: float = Field(ge=0, le=100)
    timeElapsed: float = Field(ge=0)
    delta: float = Field(gt=0, le=3600)
    temperature_C: float = Field(default=25, ge=-30, le=80)
    session_id: str | None = Field(default=None, max_length=80)
