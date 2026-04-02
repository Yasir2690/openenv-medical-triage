"""
Reward calculation helpers for Medical Triage Environment.
Core reward logic lives in MedicalTriageEnv._calculate_reward() (environment.py).
This module re-exports TriageReward for convenience and documents the reward schema.

Reward schema (all components sum to reward.total):
  patient_outcome_score  : 0.0 – 0.5  (correct ESI, reduced mortality)
  wait_time_score        : 0.0 – 0.3  (critical patients seen quickly)
  resource_score         : 0.0 – 0.2  (efficient room/doctor allocation)
  penalty                : -0.5 – 0.0 (LWBS rate, mortality events)
  total                  : sum of above, clipped to [-0.5, 1.0]
"""

from src.models import TriageReward  # noqa: F401  (re-export for convenience)

__all__ = ["TriageReward"]
