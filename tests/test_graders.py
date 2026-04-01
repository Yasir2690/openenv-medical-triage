from datetime import datetime, timedelta

from src.graders import grade_easy_task, grade_medium_task, grade_hard_task
from src.models import (
	ChiefComplaint,
	ESILevel,
	Patient,
	TriageAction,
)


def _sample_patient(complaint=ChiefComplaint.CHEST_PAIN):
	return Patient(
		id="PAT_TEST",
		arrival_time=datetime.now() - timedelta(minutes=5),
		age=55,
		chief_complaint=complaint,
		triage_note="test note",
		vital_signs={},
		conditions=[],
		allergies=[],
		medications=[],
		triage_time=datetime.now(),
	)


def test_easy_grader_returns_score_in_range():
	patient = _sample_patient(ChiefComplaint.CHEST_PAIN)
	action = TriageAction(patient_id=patient.id, esi_level=ESILevel.EMERGENT)
	history = [{"action": action, "patient": patient}]

	score = grade_easy_task(history)
	assert 0.0 <= score <= 1.0


def test_medium_grader_returns_score_in_range():
	patient = _sample_patient()
	action = TriageAction(patient_id=patient.id, esi_level=ESILevel.URGENT, assigned_room="bed_1")
	history = [
		{
			"action": action,
			"patient": patient,
			"info": {"metrics": {"total_arrivals": 30, "total_lwbs": 2}},
		}
	]

	score = grade_medium_task(history)
	assert 0.0 <= score <= 1.0


def test_hard_grader_returns_score_in_range():
	patient = _sample_patient(ChiefComplaint.UNRESPONSIVE)
	action = TriageAction(patient_id=patient.id, esi_level=ESILevel.RESUSCITATION)
	history = [
		{
			"action": action,
			"patient": patient,
			"info": {"metrics": {"total_arrivals": 45, "total_mortality": 1}},
		}
	]

	score = grade_hard_task(history)
	assert 0.0 <= score <= 1.0

