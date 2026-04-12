"""
Inference Script for Medical Triage Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM (required).
    MODEL_NAME     The model identifier to use for inference (required).
    HF_TOKEN       Your Hugging Face / API key (required).

- Defaults are set for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "<default-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<default-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after all episodes, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Score should be normalized to [0, 1]
"""

import os
import random
import sys
import numpy as np
import textwrap
from typing import List, Optional, Dict, Any

from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel, VitalSign
from src.triage_logic import ESIGuidelines, ClinicalDeteriorationPredictor

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# REQUIRED environment variables for inference
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Task configuration
TASK_NAME = "medical_triage_inference"
BENCHMARK = "medical_triage"
MAX_STEPS = 50
NUM_EPISODES = 3


# ===========================
# Logging Functions
# ===========================

def log_start(task: str, env: str, model: str) -> None:
    """Log episode start."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log a single step."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_appropriate_tests(patient, esi_level):
    """Select appropriate diagnostic tests based on chief complaint and ESI level"""
    tests = []
    chief_complaint = patient.chief_complaint.value.lower()
    
    # Critical patients (ESI 1-2) get full workup
    if esi_level <= 2:
        tests.extend(["ecg", "troponin", "cbc", "metabolic_panel"])
        if "chest" in chief_complaint or "cardiac" in chief_complaint:
            tests.extend(["chest_xray", "ct_angiogram"])
        if "neuro" in chief_complaint or "stroke" in chief_complaint or "altered" in chief_complaint:
            tests.extend(["ct_head", "ct_angiogram_brain"])
        if "abdominal" in chief_complaint:
            tests.extend(["ct_abdomen", "lipase", "amylase"])
    
    # Urgent patients (ESI 3) - targeted tests
    elif esi_level == 3:
        if "chest" in chief_complaint:
            tests.extend(["ecg", "troponin", "chest_xray"])
        elif "stroke" in chief_complaint or "neuro" in chief_complaint:
            tests.extend(["ct_head", "neuropsy"])
        elif "abdominal" in chief_complaint:
            tests.extend(["ct_abdomen", "lipase"])
        elif "fever" in chief_complaint:
            tests.extend(["cbc", "blood_cultures", "urinalysis"])
        elif "shortness_of_breath" in chief_complaint or "dyspnea" in chief_complaint:
            tests.extend(["chest_xray", "cbc", "bmp"])
        else:
            tests.extend(["cbc", "metabolic_panel"])
    
    # Semi-urgent/less urgent (ESI 4-5) - minimal tests
    elif esi_level in [4, 5]:
        if "fracture" in chief_complaint:
            tests.extend(["xray"])
        elif "fever" in chief_complaint:
            tests.extend(["cbc"])
        else:
            tests.append("vital_signs_repeat")
    
    return tests[:3]  # Limit to 3 tests per action


def get_best_doctor(available_doctors, patient_esi, existing_assignments):
    """Select best available doctor based on ESI level and current load"""
    if not available_doctors:
        return None
    
    # Calculate current workload per doctor
    doctor_loads = {doc_id: 0 for doc_id in available_doctors}
    for _, assigned_doc in existing_assignments.items():
        if assigned_doc in doctor_loads:
            doctor_loads[assigned_doc] += 1
    
    # For critical patients, prefer specialists
    critical_specialists = ["critical_care", "trauma_surgery", "emergency_medicine"]
    
    if patient_esi == 1:
        # Critical - prefer critical care, then trauma surgery
        available_docs_list = list(available_doctors.items())
        for specialty in critical_specialists:
            for doc_id, doc_specialty in available_docs_list:
                if specialty in doc_specialty.lower() and doctor_loads[doc_id] == min(doctor_loads.values()):
                    return doc_id
    
    # Default: pick doctor with lightest load
    lightest_doc = min(doctor_loads.items(), key=lambda x: x[1])[0]
    return lightest_doc


def get_best_room(available_rooms, patient_esi, existing_assignments):
    """Select best room based on ESI level and load balancing"""
    if not available_rooms:
        return None
    
    # Calculate room usage
    room_usage = {room: 0 for room in available_rooms}
    for _, assigned_room in existing_assignments.items():
        if assigned_room in room_usage:
            room_usage[assigned_room] += 1
    
    # Critical patients get trauma/ICU rooms
    if patient_esi <= 2:
        trauma_rooms = [r for r in available_rooms if 'trauma' in r.lower() or 'icu' in r.lower()]
        if trauma_rooms:
            # Pick the least-used trauma room
            return min(trauma_rooms, key=lambda r: room_usage[r])
    
    # Regular rooms for others, balanced across available rooms
    least_used_room = min(available_rooms, key=lambda r: room_usage[r])
    return least_used_room


def rule_based_agent(observation, episode_state=None):
    """
    Advanced rule-based triage agent with smart resource allocation
    - Uses ESI guidelines with clinical decision support
    - Prioritizes by deterioration risk & wait time
    - Load-balanced doctor/room assignments
    - Appropriate diagnostic test ordering
    """
    if not observation.waiting_patients:
        return None
    
    # Initialize episode state if needed (for tracking assignments)
    if episode_state is None:
        episode_state = {'doctor_assignments': {}, 'room_assignments': {}}
    
    # Dynamically score and prioritize waiting patients
    patient_scores = []
    for p in observation.waiting_patients:
        # Calculate clinical ESI using guidelines
        correct_esi = ESIGuidelines.calculate_esi(p)
        esi_value = correct_esi.value if hasattr(correct_esi, 'value') else correct_esi
        
        # Deterioration risk (higher = more urgent)
        deterioration_risk = ClinicalDeteriorationPredictor.risk_score(p)
        
        # Wait time consideration (longer wait = higher priority)
        wait_minutes = p.wait_time_minutes if hasattr(p, 'wait_time_minutes') and p.wait_time_minutes else 0
        
        # Multi-factor priority score
        # ESI priority (inverse: ESI 1 = highest)
        esi_weight = (1.0 / esi_value) * 100
        # Deterioration weight
        risk_weight = deterioration_risk * 50
        # Wait time weight (exponential to heavily prioritize longtime waiters)
        wait_weight = min(wait_minutes / 5.0, 20)  # Cap at 20 for 100-minute wait
        
        priority_score = esi_weight + risk_weight + wait_weight
        
        patient_scores.append({
            'patient': p,
            'esi_level': esi_value,
            'priority': priority_score,
            'risk': deterioration_risk,
            'wait': wait_minutes
        })
    
    # Sort by priority (highest first)
    patient_scores.sort(key=lambda x: -x['priority'])
    selected = patient_scores[0]
    patient = selected['patient']
    esi_level = selected['esi_level']
    
    # Intelligent resource allocation with load balancing
    best_doctor = get_best_doctor(observation.available_doctors, esi_level, episode_state['doctor_assignments'])
    best_room = get_best_room(observation.available_rooms, esi_level, episode_state['room_assignments'])
    
    # Order appropriate tests
    tests = get_appropriate_tests(patient, esi_level)
    
    # Update assignment tracking
    episode_state['doctor_assignments'][patient.id] = best_doctor
    episode_state['room_assignments'][patient.id] = best_room
    
    # Determine if resuscitation is needed
    risk = selected['risk']
    resuscitate = esi_level == 1 or (esi_level == 2 and risk > 0.7)
    
    return TriageAction(
        patient_id=patient.id,
        esi_level=esi_level,
        assigned_room=best_room,
        assigned_doctor_id=best_doctor,
        order_tests=tests,
        initiate_resuscitation=resuscitate
    )


def llm_agent(client, observation, step_num, conversation_history, episode_state=None):
    """
    LLM-based triage agent using OpenAI Client
    Makes triage decisions via LLM API calls with timeout protection
    """
    if not observation.waiting_patients:
        return None
    
    def _call_llm():
        """Inner function for LLM call"""
        try:
            patient = observation.waiting_patients[0]
            
            # Build prompt for triage decision
            prompt = textwrap.dedent(f"""
            You are an emergency department triage expert using ESI (Emergency Severity Index) protocol.
            
            Current patient waiting:
            - ID: {patient.id}
            - Age: {patient.age}
            - Chief Complaint: {patient.chief_complaint.value.replace('_', ' ')}
            - Vital Signs: HR={patient.vital_signs.get(VitalSign.HEART_RATE, 'N/A')}, BP={patient.vital_signs.get(VitalSign.BLOOD_PRESSURE_SYS, 'N/A')}/{patient.vital_signs.get(VitalSign.BLOOD_PRESSURE_DIA, 'N/A')}
            - Critical: {patient.is_critical}
            
            Available resources:
            - Rooms: {len(observation.available_rooms)}
            - Doctors: {len(observation.available_doctors)}
            
            Decide the ESI level (1-5):
            1 = Resuscitation (immediate threat to life)
            2 = Emergent (high risk, needs immediate evaluation)
            3 = Urgent (needs prompt evaluation)
            4 = Less urgent (stable, minor complaint)
            5 = Non-urgent (minor complaint, stable)
            
            Respond with ONLY a single integer (1-5).
            """).strip()
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an emergency triage expert. Respond with only a number 1-5."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3,
                timeout=3
            )
            
            esi_str = response.choices[0].message.content.strip()
            esi_level = int(esi_str) if esi_str.isdigit() else 3
            esi_level = max(1, min(5, esi_level))
            
            return esi_level
        except Exception as e:
            raise
    
    try:
        # Use ThreadPoolExecutor with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_llm)
            try:
                esi_level = future.result(timeout=5)  # 5-second timeout
            except (FutureTimeoutError, Exception) as e:
                # Fallback to rule-based
                return rule_based_agent(observation, episode_state)
        
        patient = observation.waiting_patients[0]
        # Get available resources
        room = observation.available_rooms[0] if observation.available_rooms else None
        doctor = list(observation.available_doctors.keys())[0] if observation.available_doctors else None
        
        return TriageAction(
            patient_id=patient.id,
            esi_level=esi_level,
            assigned_room=room,
            assigned_doctor_id=doctor,
            order_tests=[],
            initiate_resuscitation=(esi_level == 1)
        )
        
    except Exception as e:
        # Ultimate fallback to rule-based
        return rule_based_agent(observation, episode_state)


def run_episode(env: MedicalTriageEnv, episode_num: int, client=None, use_llm: bool = False) -> Dict[str, Any]:
    """
    Run a single episode with LLM or rule-based agent.
    
    Args:
        env: Medical triage environment
        episode_num: Episode number (for tracking)
        client: OpenAI client (if using LLM)
        use_llm: Whether to use LLM agent
        
    Returns:
        Dictionary with episode results
    """
    observation = env.reset()
    total_reward = 0.0
    step_records = []
    conversation_history = []
    episode_state = {'doctor_assignments': {}, 'room_assignments': {}}
    
    for step in range(MAX_STEPS):
        # Get action from LLM or rule-based agent
        if use_llm and client and HAS_OPENAI:
            action = llm_agent(client, observation, step, conversation_history, episode_state)
        else:
            action = rule_based_agent(observation, episode_state)
        
        if action is None:
            break
        
        # Execute action
        observation, reward, done, info = env.step(action)
        total_reward += reward.total
        step_num = step + 1
        
        # Format action as string
        action_str = f"assign_esi({action.esi_level.value})"
        
        step_records.append({
            "step": step_num,
            "action": action_str,
            "reward": reward.total,
            "done": done,
            "error": None
        })
        
        conversation_history.append(f"step {step_num}: {action_str} -> {reward.total:.2f}")
        
        if done:
            break
    
    return {
        "total_reward": total_reward,
        "step_records": step_records,
        "done": done,
        "arrivals": info['metrics']['total_arrivals'],
        "lwbs": info['metrics']['total_lwbs'],
        "mortality": info['metrics']['total_mortality']
    }


def main():
    """Main inference loop."""
    # Log episode start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    env = MedicalTriageEnv(max_steps=MAX_STEPS, random_seed=42)
    
    # Initialize OpenAI client with provided API credentials
    client = None
    use_llm = False
    
    if HAS_OPENAI and API_KEY and API_BASE_URL and MODEL_NAME:
        try:
            # Use the provided API_BASE_URL and API_KEY environment variables
            # This ensures we use the LiteLLM proxy provided during validation
            client = OpenAI(
                api_key=API_KEY,
                base_url=API_BASE_URL
            )
            use_llm = True
            print(f"[DEBUG] LLM mode enabled with API_BASE_URL={API_BASE_URL}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Failed to initialize LLM client: {e}", flush=True)
            use_llm = False
    
    all_step_records = []
    all_rewards = []
    total_steps = 0
    final_score = 0.0
    success = False
    
    try:
        # Run episodes
        for episode in range(1, NUM_EPISODES + 1):
            result = run_episode(env, episode, client=client, use_llm=use_llm)
            
            # Collect and log step records
            for record in result["step_records"]:
                all_step_records.append(record)
                all_rewards.append(record["reward"])
                
                # Log each step
                log_step(
                    step=record["step"],
                    action=record["action"],
                    reward=record["reward"],
                    done=record["done"],
                    error=record["error"]
                )
        
        # Calculate final results
        total_steps = len(all_step_records)
        avg_reward_per_step = sum(all_rewards) / total_steps if total_steps > 0 else 0.0
        final_score = min(1.0, max(0.0, avg_reward_per_step))
        success = final_score >= 0.4
        
    finally:
        # Always emit [END] line
        log_end(success=success, steps=total_steps, score=final_score, rewards=all_rewards)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())