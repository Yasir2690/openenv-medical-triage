"""Clinical decision support logic for ESI triage"""

from .models import Patient, ChiefComplaint, ESILevel


class ESIGuidelines:
    """Implementation of ESI triage algorithm"""
    
    @classmethod
    def calculate_esi(cls, patient: Patient) -> ESILevel:
        """Calculate ESI level based on clinical presentation"""
        
        if patient.chief_complaint in [
            ChiefComplaint.UNRESPONSIVE,
            ChiefComplaint.SEVERE_BLEEDING,
            ChiefComplaint.SEIZURE
        ]:
            return ESILevel.RESUSCITATION
        
        if patient.chief_complaint in [
            ChiefComplaint.CHEST_PAIN,
            ChiefComplaint.STROKE_SYMPTOMS,
            ChiefComplaint.HEAD_INJURY,
            ChiefComplaint.ALTERED_MENTAL_STATUS
        ]:
            return ESILevel.EMERGENT
        
        if patient.chief_complaint in [
            ChiefComplaint.ABDOMINAL_PAIN,
            ChiefComplaint.SHORTNESS_OF_BREATH,
            ChiefComplaint.FEVER
        ]:
            return ESILevel.URGENT
        
        if patient.chief_complaint == ChiefComplaint.FRACTURE:
            return ESILevel.SEMI_URGENT
        
        return ESILevel.NON_URGENT


class ClinicalDeteriorationPredictor:
    """Predict patient deterioration"""
    
    @staticmethod
    def risk_score(patient: Patient) -> float:
        """Calculate deterioration risk score (0-1)"""
        score = 0.0
        
        if patient.age > 75:
            score += 0.3
        elif patient.age > 65:
            score += 0.2
        elif patient.age > 50:
            score += 0.1
        
        if patient.chief_complaint in [
            ChiefComplaint.CHEST_PAIN,
            ChiefComplaint.STROKE_SYMPTOMS,
            ChiefComplaint.SHORTNESS_OF_BREATH
        ]:
            score += 0.2
        
        if patient.assigned_esi in [ESILevel.RESUSCITATION, ESILevel.EMERGENT]:
            score += 0.2
        
        return min(score, 1.0)
#hi#