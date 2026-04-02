"""
Inference Script for Medical Triage Environment
Runs 3 task-specific episodes, grades each, and prints per-task scores.
Supports optional LLM-based agent via OpenAI client (HF_TOKEN + API_BASE_URL).
"""

import os
import random
import json
import numpy as np

from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel
from src.graders import grade_easy_task, grade_medium_task, grade_hard_task
from src.triage_logic import ESIGuidelines

# ── Environment variables for LLM agent (optional) ──────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

# ── Task configuration ───────────────────────────────────────────────────────
TASK_CONFIG = {
    "easy": {
        "max_steps": 30,
        "max_patients": 20,
        "random_seed": 42,
        "description": "Basic ESI Triage Accuracy",
        "grader": grade_easy_task,
    },
    "medium": {
        "max_steps": 60,
        "max_patients": 40,
        "random_seed": 7,
        "description": "Resource Allocation Under Pressure",
        "grader": grade_medium_task,
    },
    "hard": {
        "max_steps": 100,
        "max_patients": 60,
        "random_seed": 99,
        "description": "Mass Casualty Incident",
        "grader": grade_hard_task,
    },
}


# ── Optional: OpenAI / HF Inference API client ──────────────────────────────
def get_llm_client():
    """
    Returns an OpenAI-compatible client if HF_TOKEN or OPENAI_API_KEY is set.
    Uses HuggingFace Inference API by default; set API_BASE_URL for other endpoints.
    """
    if not API_KEY:
        return None
    try:
        import openai
        client = openai.OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        return client
    except Exception as e:
        print(f"[WARN] Could not initialise OpenAI client: {e}")
        return None


def llm_triage_agent(observation, client):
    """
    LLM-based triage agent.  Sends a compact observation summary to the model
    and parses a structured JSON response back into a TriageAction.
    Falls back to rule-based agent on any error.
    """
    if not observation.waiting_patients:
        return None

    patient = observation.waiting_patients[0]

    prompt = (
        "You are an emergency triage nurse. Given the patient details below, "
        "decide the ESI level (1=resuscitation, 2=emergent, 3=urgent, 4=semi-urgent, 5=non-urgent) "
        "and optionally assign a room/doctor. Reply ONLY with valid JSON like: "
        '{"esi_level": 3, "initiate_resuscitation": false}\n\n'
        f"Patient ID: {patient.id}\n"
        f"Age: {patient.age}\n"
        f"Chief complaint: {patient.chief_complaint.value}\n"
        f"Triage note: {patient.triage_note}\n"
        f"Vital signs: {json.dumps({k.value: v for k, v in patient.vital_signs.items()}, default=str)}\n"
        f"Conditions: {patient.conditions}\n"
        f"Available rooms: {observation.available_rooms[:3]}\n"
        f"Available doctors: {list(observation.available_doctors.keys())[:3]}\n"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        esi_level = int(data.get("esi_level", 3))
        esi_level = max(1, min(5, esi_level))

        room = observation.available_rooms[0] if observation.available_rooms else None
        doctor = list(observation.available_doctors.keys())[0] if observation.available_doctors else None

        return TriageAction(
            patient_id=patient.id,
            esi_level=esi_level,
            assigned_room=room,
            assigned_doctor_id=doctor,
            order_tests=[],
            initiate_resuscitation=data.get("initiate_resuscitation", esi_level == 1),
        )
    except Exception as e:
        print(f"  [LLM fallback] {e}")
        return rule_based_agent(observation)


# ── Rule-based agent (always-available fallback) ─────────────────────────────
def rule_based_agent(observation):
    """
    Classic ESI-guideline rule-based triage agent.
    Prioritises critical patients, then by clinical acuity.
    """
    if not observation.waiting_patients:
        return None

    # Sort: critical (ESI 1) > emergent (ESI 2) > others
    critical = [p for p in observation.waiting_patients
                if p.chief_complaint.value in ("unresponsive", "severe_bleeding", "seizure")]
    emergent = [p for p in observation.waiting_patients
                if p.chief_complaint.value in ("chest_pain", "stroke_symptoms", "head_injury",
                                               "altered_mental_status")
                and p not in critical]
    remaining = [p for p in observation.waiting_patients
                 if p not in critical and p not in emergent]

    if critical:
        patient = critical[0]
        esi_level = ESILevel.RESUSCITATION
    elif emergent:
        patient = emergent[0]
        esi_level = ESILevel.EMERGENT
    elif remaining:
        patient = remaining[0]
        # Use clinical guidelines to decide ESI
        esi_level = ESIGuidelines.calculate_esi(patient)
    else:
        return None

    room = observation.available_rooms[0] if observation.available_rooms else None
    doctor = list(observation.available_doctors.keys())[0] if observation.available_doctors else None

    return TriageAction(
        patient_id=patient.id,
        esi_level=int(esi_level),
        assigned_room=room,
        assigned_doctor_id=doctor,
        order_tests=[],
        initiate_resuscitation=(int(esi_level) == 1),
    )


# ── Episode runner ────────────────────────────────────────────────────────────
def run_task_episode(task_name, config, llm_client=None):
    """
    Runs a single task episode and returns (grader_score, summary_dict).
    Collects full episode history for the grader.
    """
    print(f"\n{'─'*60}")
    print(f"  TASK: {task_name.upper()} — {config['description']}")
    print(f"{'─'*60}")

    env = MedicalTriageEnv(
        max_steps=config["max_steps"],
        max_patients=config["max_patients"],
        random_seed=config["random_seed"],
    )

    observation = env.reset()
    episode_history = []
    total_reward = 0.0
    step_count = 0
    agent_mode = "LLM" if llm_client else "Rule-based"
    print(f"  Agent mode: {agent_mode}")

    for step in range(config["max_steps"]):
        # Choose action
        if llm_client:
            action = llm_triage_agent(observation, llm_client)
        else:
            action = rule_based_agent(observation)

        if action is None:
            break

        # Capture the patient being acted on (for graders)
        patient = env.patients.get(action.patient_id)

        # Execute
        observation, reward, done, info = env.step(action)
        total_reward += reward.total
        step_count = step + 1

        # Log step for grader
        episode_history.append({
            "step": step,
            "action": action,
            "patient": patient,
            "reward": reward,
            "info": info,
        })

        if step % 10 == 0:
            print(f"  Step {step:3d} | reward={reward.total:.3f} | "
                  f"LWBS={observation.lwbs_rate:.1%} | "
                  f"active_patients={info['active_patients']}")

        if done:
            print(f"  Episode ended at step {step + 1}")
            break

    # Grade
    grader_score = config["grader"](episode_history)

    summary = {
        "task": task_name,
        "agent": agent_mode,
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        "grader_score": grader_score,
        "arrivals": info["metrics"]["total_arrivals"],
        "lwbs": info["metrics"]["total_lwbs"],
        "mortality": info["metrics"]["total_mortality"],
    }

    print(f"\n  Total reward  : {total_reward:.4f}")
    print(f"  Grader score  : {grader_score:.4f}  (0.0 – 1.0)")
    print(f"  Patients seen : {summary['arrivals']}")
    print(f"  LWBS          : {summary['lwbs']}")
    print(f"  Mortality     : {summary['mortality']}")

    return grader_score, summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Medical Triage Environment — OpenEnv Submission")
    print("=" * 60)

    llm_client = get_llm_client()
    if llm_client:
        print(f"\n[INFO] LLM agent active: {MODEL_NAME} via {API_BASE_URL}")
    else:
        print("\n[INFO] No API key found — using rule-based baseline agent.")

    all_scores = {}
    all_summaries = []

    for task_name, config in TASK_CONFIG.items():
        score, summary = run_task_episode(task_name, config, llm_client)
        all_scores[task_name] = score
        all_summaries.append(summary)

    # ── Final per-task score report ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL GRADER SCORES (0.0 – 1.0)")
    print("=" * 60)
    targets = {"easy": 0.70, "medium": 0.60, "hard": 0.50}
    for task_name, score in all_scores.items():
        target = targets[task_name]
        status = "✓" if score >= target else "✗"
        print(f"  {status}  {task_name:<8}  score={score:.4f}  target={target:.2f}")

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"\n  Average grader score: {avg:.4f}")
    print("=" * 60)
    print("\n[OK] Inference complete — environment is submission-ready.")


if __name__ == "__main__":
    main()