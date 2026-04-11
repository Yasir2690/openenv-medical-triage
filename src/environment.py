"""
OpenEnv Medical Triage Environment
Implements step(), reset(), state() API for RL agent training
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import numpy as np

from .models import (
    Patient, TriageAction, TriageObservation, TriageReward,
    ESILevel, ResourceType, DoctorSpecialty, VitalSign
)
from .triage_logic import ESIGuidelines, ClinicalDeteriorationPredictor
from .simulation import PatientGenerator, ResourceManager


class MedicalTriageEnv:
    """
    Emergency Department Triage Environment
    
    An AI agent acts as triage coordinator, assigning ESI levels,
    allocating resources, and managing patient flow.
    """
    
    def __init__(
        self,
        max_steps: int = 100,
        max_patients: int = 50,
        simulation_speed: float = 1.0,
        random_seed: Optional[int] = None,
        enable_deterioration: bool = True
    ):
        self.max_steps = max_steps
        self.max_patients = max_patients
        self.simulation_speed = simulation_speed
        self.enable_deterioration = enable_deterioration
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.patient_generator = PatientGenerator(random_seed=random_seed)
        self.resource_manager = ResourceManager()
        self.deterioration_predictor = ClinicalDeteriorationPredictor()
        
        self.step_count = 0
        self.current_time = datetime.now()
        self.patients: Dict[str, Patient] = {}
        self.patient_queue: deque = deque()
        
        self.metrics = {
            "total_arrivals": 0,
            "total_discharged": 0,
            "total_lwbs": 0,
            "total_mortality": 0,
            "critical_wait_times": [],
            "esi_accuracy": [],
        }
        
        self.episode_start_time = None
        self.last_action_reward = 0.0
    
    def reset(self) -> TriageObservation:
        """Reset environment to initial state"""
        self.step_count = 0
        self.current_time = datetime.now()
        self.patients = {}
        self.patient_queue = deque()
        
        self.metrics = {
            "total_arrivals": 0,
            "total_discharged": 0,
            "total_lwbs": 0,
            "total_mortality": 0,
            "critical_wait_times": [],
            "esi_accuracy": [],
        }
        
        self.episode_start_time = self.current_time
        self.last_action_reward = 0.0
        
        initial_patients = self.patient_generator.generate_batch(
            count=random.randint(5, 10),
            current_time=self.current_time
        )
        for patient in initial_patients:
            self._add_patient(patient)
        
        return self._get_observation()
    
    def step(self, action: TriageAction) -> Tuple[TriageObservation, TriageReward, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        is_valid = self._validate_action(action)
        self.last_action_reward = self._apply_action(action)
        
        self._advance_time()
        self._process_arrivals()
        self._update_patient_statuses()
        done = self._is_episode_done()
        
        reward = self._calculate_reward(action)
        observation = self._get_observation()
        
        info = {
            "step": self.step_count,
            "current_time": self.current_time,
            "active_patients": len(self.patients),
            "metrics": self.metrics,
            "action_valid": is_valid,
            "action_reward": self.last_action_reward,
        }
        
        return observation, reward, done, info
    
    def state(self) -> Dict[str, Any]:
        """Return current internal state"""
        return {
            "step_count": self.step_count,
            "current_time": self.current_time.isoformat(),
            "patients": {pid: patient.model_dump() for pid, patient in self.patients.items()},
            "patient_queue": list(self.patient_queue),
            "metrics": self.metrics,
            "episode_start_time": self.episode_start_time.isoformat() if self.episode_start_time else None,
            "last_action_reward": self.last_action_reward,
        }
    
    def _add_patient(self, patient: Patient) -> None:
        """Add patient to environment"""
        self.patients[patient.id] = patient
        self.patient_queue.append(patient.id)
        self.metrics["total_arrivals"] += 1
    
    def _validate_action(self, action: TriageAction) -> bool:
        """Validate action"""
        if action.patient_id not in self.patients:
            return False
        patient = self.patients[action.patient_id]
        if patient.discharged_time is not None or patient.left_without_being_seen or patient.mortality:
            return False
        return True
    
    def _apply_action(self, action: TriageAction) -> float:
        """Apply action effects and return immediate reward"""
        if action.patient_id not in self.patients:
            return -0.5
        
        patient = self.patients[action.patient_id]
        reward = 0.0
        
        # ESI assignment
        if action.esi_level:
            correct_esi = ESIGuidelines.calculate_esi(patient)
            
            if action.esi_level == correct_esi:
                reward += 0.3
                patient.assigned_esi = action.esi_level
                patient.triage_time = self.current_time
                self.metrics["esi_accuracy"].append(1)
            else:
                penalty = -0.2 if correct_esi in [1, 2] else -0.1
                reward += penalty
                patient.assigned_esi = action.esi_level
                patient.triage_time = self.current_time
                self.metrics["esi_accuracy"].append(0)
        
        # Resource allocation
        if action.assigned_room:
            if action.assigned_room in self.resource_manager.available_rooms:
                if self.resource_manager.assign_room(action.assigned_room):
                    patient.assigned_room = action.assigned_room
                    reward += 0.1
        
        if action.assigned_doctor_id:
            if action.assigned_doctor_id in self.resource_manager.available_doctors:
                if self.resource_manager.assign_doctor(action.assigned_doctor_id):
                    patient.assigned_doctor_id = action.assigned_doctor_id
                    patient.seen_time = self.current_time
                    reward += 0.2
                    
                    wait_time = (patient.seen_time - patient.arrival_time).total_seconds() / 60
                    if patient.assigned_esi in [1, 2]:
                        self.metrics["critical_wait_times"].append(wait_time)
                        # BONUS: Fast triage of critical patients
                        if wait_time < 15:
                            reward += 0.05
                        # BONUS: Prevented deterioration by seeing patient quickly
                        if not patient.has_deteriorated:
                            reward += 0.05  # Reward for prevented deterioration
        
        # STEP-LEVEL BONUS: Maintaining low LWBS rate demonstrates good overall performance
        if self.metrics["total_arrivals"] > 0:
            current_lwbs = self.metrics["total_lwbs"] / self.metrics["total_arrivals"]
            if current_lwbs < 0.02:  # If LWBS < 2%, small bonus
                reward += 0.02
        
        return max(-1.0, min(1.0, reward))
    
    def _calculate_reward(self, action: TriageAction) -> TriageReward:
        """Calculate comprehensive reward"""
        patient_outcome = 0.3
        wait_time_score = 0.15
        resource_score = 0.1
        
        # LWBS penalty
        lwbs_rate = self.metrics["total_lwbs"] / max(1, self.metrics["total_arrivals"])
        penalty = 0.0
        if lwbs_rate > 0.05:
            penalty -= 0.1 * (lwbs_rate / 0.05)
        
        # Mortality penalty
        if self.metrics["total_mortality"] > 0:
            penalty -= 0.2 * self.metrics["total_mortality"]
        
        penalty += self.last_action_reward * 0.1
        penalty = max(-0.5, min(0, penalty))
        
        return TriageReward(
            patient_outcome_score=min(0.5, patient_outcome),
            wait_time_score=min(0.3, wait_time_score),
            resource_score=min(0.2, resource_score),
            penalty=penalty
        )
    
    def _advance_time(self) -> None:
        """Advance simulation time"""
        self.current_time += timedelta(minutes=self.simulation_speed)
    
    def _process_arrivals(self) -> None:
        """Generate new patient arrivals"""
        arrival_rate = 8.0 / 60
        expected_arrivals = arrival_rate * self.simulation_speed
        num_arrivals = np.random.poisson(expected_arrivals)
        
        for _ in range(min(num_arrivals, 3)):
            if len(self.patients) < self.max_patients:
                new_patient = self.patient_generator.generate_patient(
                    current_time=self.current_time
                )
                self._add_patient(new_patient)
    
    def _update_patient_statuses(self) -> None:
        """Update patient outcomes and detect deterioration"""
        to_remove = []
        
        for patient_id, patient in self.patients.items():
            wait_time = patient.wait_time_minutes
            
            # DETERIORATION CHECK: If patient has high acuity and has been waiting, they may deteriorate
            if patient.assigned_esi in [1, 2] and not patient.seen_time:
                # High-risk critical patients deteriorate faster
                deterioration_threshold = 10 if patient.assigned_esi == 1 else 25
                if wait_time and wait_time > deterioration_threshold and not patient.has_deteriorated:
                    patient.has_deteriorated = True
                    patient.deterioration_risk = 0.8
                    patient.deterioration_events.append({
                        'time': self.current_time,
                        'event': f'Patient deteriorated after {wait_time:.0f} min wait',
                        'severity': 'critical'
                    })
                    # Deterioration increases mortality risk
                    if np.random.random() < 0.15:  # 15% chance of mortality if deteriorated
                        patient.mortality = True
                        self.metrics["total_mortality"] += 1
            
            # LWBS
            if not patient.seen_time and not patient.discharged_time and not patient.left_without_being_seen:
                threshold = {
                    1: 10, 2: 30, 3: 60, 4: 120, 5: 180
                }.get(patient.assigned_esi, 60) if patient.assigned_esi else 60
                
                if wait_time and wait_time > threshold:
                    patient.left_without_being_seen = True
                    self.metrics["total_lwbs"] += 1
                    to_remove.append(patient_id)
            
            # Discharge
            elif patient.seen_time and not patient.discharged_time:
                treatment_time = (self.current_time - patient.seen_time).total_seconds() / 60
                if treatment_time > 30:
                    patient.discharged_time = self.current_time
                    self.metrics["total_discharged"] += 1
                    to_remove.append(patient_id)
                    
                    if patient.assigned_room:
                        self.resource_manager.rooms[patient.assigned_room]["available"] = True
                    if patient.assigned_doctor_id:
                        self.resource_manager.doctors[patient.assigned_doctor_id]["available"] = True
        
        for patient_id in to_remove:
            if patient_id in self.patients:
                del self.patients[patient_id]
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        if self.step_count >= self.max_steps:
            return True
        
        if len(self.patients) == 0 and self.metrics["total_arrivals"] > 10:
            return True
        
        lwbs_rate = self.metrics["total_lwbs"] / max(1, self.metrics["total_arrivals"])
        if lwbs_rate > 0.20:
            return True
        
        if self.metrics["total_mortality"] > 3:
            return True
        
        return False
    
    def _get_observation(self) -> TriageObservation:
        """Build observation from current state"""
        waiting = []
        triaged = []
        active = []
        
        for patient in self.patients.values():
            if patient.discharged_time:
                continue
            elif not patient.triage_time:
                waiting.append(patient)
            elif patient.triage_time and not patient.seen_time:
                triaged.append(patient)
            else:
                active.append(patient)
        
        wait_times = {}
        for esi in [1, 2, 3, 4, 5]:
            esi_patients = [p for p in waiting + triaged if p.assigned_esi == esi]
            if esi_patients:
                waits = [p.wait_time_minutes for p in esi_patients if p.wait_time_minutes]
                if waits:
                    wait_times[esi] = np.mean(waits)
        
        return TriageObservation(
            timestamp=self.current_time,
            waiting_patients=waiting[:10],
            triaged_patients=triaged[:10],
            active_patients=active[:10],
            available_rooms=self.resource_manager.available_rooms,
            available_doctors=self.resource_manager.available_doctors,
            equipment_available=self.resource_manager.equipment,
            current_wait_times=wait_times,
            lwbs_rate=self.metrics["total_lwbs"] / max(1, self.metrics["total_arrivals"]),
            code_blue_active=False,
            episode_step=self.step_count,
            max_steps=self.max_steps
        )