"""Pydantic models for OpenEnv Medical Triage Environment"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ESILevel(int, Enum):
    RESUSCITATION = 1
    EMERGENT = 2
    URGENT = 3
    SEMI_URGENT = 4
    NON_URGENT = 5


class VitalSign(str, Enum):
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE_SYS = "bp_systolic"
    BLOOD_PRESSURE_DIA = "bp_diastolic"
    RESPIRATORY_RATE = "respiratory_rate"
    OXYGEN_SATURATION = "o2_sat"
    TEMPERATURE = "temperature"
    GCS = "gcs"


class ChiefComplaint(str, Enum):
    CHEST_PAIN = "chest_pain"
    SHORTNESS_OF_BREATH = "shortness_of_breath"
    HEAD_INJURY = "head_injury"
    STROKE_SYMPTOMS = "stroke_symptoms"
    SEVERE_BLEEDING = "severe_bleeding"
    ALTERED_MENTAL_STATUS = "altered_mental_status"
    ABDOMINAL_PAIN = "abdominal_pain"
    FEVER = "fever"
    FRACTURE = "fracture"
    BURN = "burn"
    ALLERGIC_REACTION = "allergic_reaction"
    SEIZURE = "seizure"
    UNRESPONSIVE = "unresponsive"
    OTHER = "other"


class ResourceType(str, Enum):
    BED = "bed"
    CT_SCANNER = "ct_scanner"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    TRAUMA_ROOM = "trauma_room"
    CARDIAC_MONITOR = "cardiac_monitor"
    VENTILATOR = "ventilator"


class DoctorSpecialty(str, Enum):
    EMERGENCY = "emergency_medicine"
    TRAUMA = "trauma_surgery"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    PEDIATRICS = "pediatrics"
    CRITICAL_CARE = "critical_care"


class Patient(BaseModel):
    id: str
    arrival_time: datetime
    age: int = Field(..., ge=0, le=120)
    chief_complaint: ChiefComplaint
    triage_note: str = ""
    vital_signs: Dict[VitalSign, Optional[float]] = Field(default_factory=dict)
    conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    assigned_esi: Optional[ESILevel] = None
    assigned_room: Optional[str] = None
    assigned_doctor_id: Optional[str] = None
    ordered_tests: List[str] = Field(default_factory=list)
    triage_time: Optional[datetime] = None
    seen_time: Optional[datetime] = None
    discharged_time: Optional[datetime] = None
    admitted: bool = False
    left_without_being_seen: bool = False
    mortality: bool = False
    deterioration_events: List[Dict] = Field(default_factory=list)
    deterioration_risk: float = 0.0  # 0.0-1.0: risk of deterioration if untreated
    has_deteriorated: bool = False  # True if patient condition worsened due to wait

    def get_wait_time_minutes(self):
        if self.seen_time and self.arrival_time:
            return (self.seen_time - self.arrival_time).total_seconds() / 60
        elif self.arrival_time:
            now = datetime.now()
            return (now - self.arrival_time).total_seconds() / 60
        return None

    @property
    def wait_time_minutes(self):
        return self.get_wait_time_minutes()

    @property
    def is_critical(self):
        if self.assigned_esi:
            return self.assigned_esi in [ESILevel.RESUSCITATION, ESILevel.EMERGENT]
        if self.chief_complaint in [ChiefComplaint.CHEST_PAIN, ChiefComplaint.STROKE_SYMPTOMS, ChiefComplaint.UNRESPONSIVE, ChiefComplaint.SEVERE_BLEEDING]:
            return True
        return False


class TriageAction(BaseModel):
    patient_id: str
    esi_level: Optional[ESILevel] = None
    assigned_room: Optional[str] = None
    assigned_doctor_id: Optional[str] = None
    order_tests: List[str] = Field(default_factory=list)
    escalate_to_specialist: bool = False
    expected_wait_time_minutes: Optional[int] = None
    initiate_resuscitation: bool = False


class TriageObservation(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    waiting_patients: List[Patient] = Field(default_factory=list)
    triaged_patients: List[Patient] = Field(default_factory=list)
    active_patients: List[Patient] = Field(default_factory=list)
    available_rooms: List[str] = Field(default_factory=list)
    available_doctors: Dict[str, DoctorSpecialty] = Field(default_factory=dict)
    equipment_available: Dict[ResourceType, int] = Field(default_factory=dict)
    current_wait_times: Dict[int, float] = Field(default_factory=dict)
    lwbs_rate: float = 0.0
    code_blue_active: bool = False
    incoming_ambulances: List[Dict] = Field(default_factory=list)
    episode_step: int = 0
    max_steps: int = 100

    @property
    def total_patients(self):
        return len(self.waiting_patients) + len(self.triaged_patients) + len(self.active_patients)


class TriageReward(BaseModel):
    patient_outcome_score: float = Field(0.0, ge=0.0, le=0.5)
    wait_time_score: float = Field(0.0, ge=0.0, le=0.3)
    resource_score: float = Field(0.0, ge=0.0, le=0.2)
    penalty: float = Field(0.0, le=0.0)

    @property
    def total(self):
        total = self.patient_outcome_score + self.wait_time_score + self.resource_score + self.penalty
        return max(0.0, min(1.0, total))
