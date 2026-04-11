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
    Easy Task: Basic ESI Triage Accuracy
    Scores based on correct ESI assignments compared to ESIGuidelines.calculate_esi()
    """
    if not episode_history:
        return 0.0

    correct = 0
    total = 0
    timely = 0

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

    if total == 0:
        return 0.0

    accuracy = (correct / total) * 0.7
    timeliness = (timely / total) * 0.3

    return round(min(1.0, accuracy + timeliness), 4)


def grade_medium_task(episode_history):
    """
    Medium Task: Resource Allocation Under Pressure
    Scores based on LWBS rate and resource utilisation across the episode.
    """
    if not episode_history:
        return 0.0

    total_patients = 0
    total_lwbs = 0
    rooms_assigned = 0
    doctors_assigned = 0
    total_actions = 0

    for step in episode_history:
        info = step.get('info', {})
        metrics = info.get('metrics', {})
        action = step.get('action')

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

    if total_patients == 0:
        return 0.0

    # LWBS rate score (lower LWBS → higher score)
    lwbs_rate = total_lwbs / total_patients
    lwbs_score = max(0.0, 1.0 - lwbs_rate * 10) * 0.5

    # Resource utilisation bonus
    if total_actions > 0:
        resource_utilisation = (rooms_assigned + doctors_assigned) / (total_actions * 2)
        resource_score = min(0.35, resource_utilisation * 0.35)
    else:
        resource_score = 0.0

    # Throughput bonus: at least 25 patients seen
    throughput_bonus = 0.15 if total_patients >= 25 else (total_patients / 25) * 0.15

    return round(min(1.0, lwbs_score + resource_score + throughput_bonus), 4)


def grade_hard_task(episode_history):
    """
    Hard Task: Mass Casualty Incident
    Scores based on mortality rate, critical patient triage speed, and surge capacity.
    Challenging: requires excellent critical patient prioritization AND resource management.
    """
    if not episode_history:
        return 0.0

    total_patients = 0
    total_mortality = 0
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
                
                # Track critical patient wait times for scoring
                if patient.triage_time and patient.arrival_time:
                    wait = (patient.triage_time - patient.arrival_time).total_seconds() / 60
                    critical_wait_times.append(wait)

    if total_patients == 0:
        return 0.0

    # STRICTER Mortality score: mortality is critical in MCI situations
    mortality_rate = total_mortality / total_patients
    mortality_score = max(0.0, 1.0 - mortality_rate * 25) * 0.45  # Increased weight

    # Critical patient triage accuracy (must be high)
    if total_critical > 0:
        critical_accuracy = (correct_critical / total_critical) * 0.35  # Increased weight
        # BONUS: critical patients seen quickly (under 15 min for ESI 1-2)
        if critical_wait_times:
            acceptable_waits = sum(1 for w in critical_wait_times if w < 15)
            speed_bonus = (acceptable_waits / len(critical_wait_times)) * 0.1
        else:
            speed_bonus = 0.0
    else:
        critical_accuracy = 0.15
        speed_bonus = 0.0

    # Stricter volume requirement for full bonus (handle true surge)
    volume_bonus = 0.1 if total_patients >= 50 else (total_patients / 50) * 0.1

    return round(min(1.0, mortality_score + critical_accuracy + speed_bonus + volume_bonus), 4)