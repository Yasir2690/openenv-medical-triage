"""
Task graders for Medical Triage Environment
Each grader returns a score between 0.0 and 1.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.triage_logic import ESIGuidelines


def grade_easy_task(episode_history):
    """
    Easy Task: Basic ESI Triage Accuracy + Timeliness
    Scores based on correct ESI assignments and avoiding patient deterioration.
    """
    if not episode_history:
        return 0.0

    correct = 0
    total = 0
    timely = 0
    total_deteriorations = 0

    for step in episode_history:
        action = step.get('action')
        patient = step.get('patient')

        if action is None or patient is None:
            continue

        if not hasattr(action, 'esi_level') or action.esi_level is None:
            continue

        total += 1
        correct_esi = ESIGuidelines.calculate_esi(patient)

        if action.esi_level == correct_esi:
            correct += 1

        # Check timeliness: patient should be seen within ESI-appropriate window
        if patient.triage_time is not None:
            wait = (patient.triage_time - patient.arrival_time).total_seconds() / 60
            thresholds = {1: 0, 2: 15, 3: 30, 4: 60, 5: 120}
            limit = thresholds.get(int(correct_esi), 60)
            if wait <= limit:
                timely += 1
        
        # Track deteriorations (penalty for missing critical patients)
        if getattr(patient, 'has_deteriorated', False):
            total_deteriorations += 1

    if total == 0:
        return 0.0

    accuracy = (correct / total) * 0.70
    timeliness = (timely / total) * 0.20
    
    # Light deterioration penalty (basic task doesn't emphasize prevention as much)
    if total > 0:
        deterioration_penalty = (total_deteriorations / total) * 0.10
        deterioration_score = max(0.0, 1.0 - deterioration_penalty) * 0.10
    else:
        deterioration_score = 0.0

    return round(min(1.0, accuracy + timeliness + deterioration_score), 4)


def grade_medium_task(episode_history):
    """
    Medium Task: Resource Allocation + Deterioration Prevention Under Pressure
    Scores based on LWBS rate, resource utilisation, and preventing patient deterioration.
    """
    if not episode_history:
        return 0.0

    total_patients = 0
    total_lwbs = 0
    total_critical = 0
    total_deteriorations = 0
    rooms_assigned = 0
    doctors_assigned = 0
    total_actions = 0

    for step in episode_history:
        info = step.get('info', {})
        metrics = info.get('metrics', {})
        action = step.get('action')
        patient = step.get('patient')

        # Take the last reported cumulative metrics
        if metrics.get('total_arrivals', 0) > total_patients:
            total_patients = metrics['total_arrivals']
            total_lwbs = metrics.get('total_lwbs', 0)

        if action is not None:
            total_actions += 1
            if getattr(action, 'assigned_room', None):
                rooms_assigned += 1
            if getattr(action, 'assigned_doctor_id', None):
                doctors_assigned += 1
        
        # Track critical patient deteriorations
        if patient is not None:
            from src.triage_logic import ESIGuidelines
            correct_esi = ESIGuidelines.calculate_esi(patient)
            if int(correct_esi) in (1, 2):
                total_critical += 1
                if getattr(patient, 'has_deteriorated', False):
                    total_deteriorations += 1

    if total_patients == 0:
        return 0.0

    # LWBS rate score (lower LWBS → higher score)
    lwbs_rate = total_lwbs / total_patients
    lwbs_score = max(0.0, 1.0 - lwbs_rate * 10) * 0.40

    # Resource utilisation bonus
    if total_actions > 0:
        resource_utilisation = (rooms_assigned + doctors_assigned) / (total_actions * 2)
        resource_score = min(0.30, resource_utilisation * 0.30)
    else:
        resource_score = 0.0

    # DETERIORATION PREVENTION SCORE: prevent critical patients from deteriorating
    if total_critical > 0:
        deterioration_rate = total_deteriorations / total_critical
        deterioration_score = max(0.0, 1.0 - deterioration_rate) * 0.20
    else:
        deterioration_score = 0.0

    # Throughput bonus: at least 25 patients seen
    throughput_bonus = 0.10 if total_patients >= 25 else (total_patients / 25) * 0.10

    return round(min(1.0, lwbs_score + resource_score + deterioration_score + throughput_bonus), 4)


def grade_hard_task(episode_history):
    """
    Hard Task: Mass Casualty Incident + Deterioration Prevention
    Scores based on mortality rate, critical patient triage speed, surge capacity,
    and ability to prevent patient deterioration under time pressure.
    """
    if not episode_history:
        return 0.0

    total_patients = 0
    total_mortality = 0
    total_deteriorations = 0
    correct_critical = 0
    total_critical = 0
    critical_wait_times = []

    for step in episode_history:
        info = step.get('info', {})
        metrics = info.get('metrics', {})
        action = step.get('action')
        patient = step.get('patient')

        if metrics.get('total_arrivals', 0) > total_patients:
            total_patients = metrics['total_arrivals']
            total_mortality = metrics.get('total_mortality', 0)

        # Check correct triage of critical patients (ESI 1–2)
        if action is not None and patient is not None:
            correct_esi = ESIGuidelines.calculate_esi(patient)
            if int(correct_esi) in (1, 2):
                total_critical += 1
                if getattr(action, 'esi_level', None) == correct_esi:
                    correct_critical += 1
                
                # Track deteriorations prevented (critical patients seen before deterioration)
                if patient.triage_time and patient.arrival_time:
                    wait = (patient.triage_time - patient.arrival_time).total_seconds() / 60
                    critical_wait_times.append(wait)
                
                # Count deteriorations that occurred
                if getattr(patient, 'has_deteriorated', False):
                    total_deteriorations += 1

    if total_patients == 0:
        return 0.0

    # Mortality score: Critical in mass casualty
    mortality_rate = total_mortality / total_patients
    mortality_score = max(0.0, 1.0 - mortality_rate * 25) * 0.40

    # Critical patient triage: Must be excellent  
    if total_critical > 0:
        critical_accuracy = (correct_critical / total_critical) * 0.30
        
        # Speed bonus: rapid critical response
        if critical_wait_times:
            acceptable_waits = sum(1 for w in critical_wait_times if w < 15)
            speed_bonus = (acceptable_waits / len(critical_wait_times)) * 0.10
        else:
            speed_bonus = 0.0
        
        # DETERIORATION PREVENTION BONUS: most important in MCI
        deterioration_rate = total_deteriorations / total_critical
        prevention_score = max(0.0, 1.0 - deterioration_rate) * 0.15
    else:
        critical_accuracy = 0.15
        speed_bonus = 0.0
        prevention_score = 0.0

    # Volume handling
    volume_bonus = 0.05 if total_patients >= 50 else (total_patients / 50) * 0.05

    return round(min(1.0, mortality_score + critical_accuracy + speed_bonus + prevention_score + volume_bonus), 4)