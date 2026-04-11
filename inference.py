"""
Inference Script for Medical Triage Environment
Rule-based and LLM-based agent with structured output format

MANDATORY REQUIREMENTS:
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN (required)
- Output format: [START], [STEP], [END] lines to stdout with flush=True
- Score normalized to [0, 1]
"""

import os
import random
import sys
import numpy as np
import textwrap
from typing import List, Optional
from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# REQUIRED environment variables for LLM inference
# These must be explicitly set in your environment configuration
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Validate required environment variables
def validate_env_vars():
    """Validate that required environment variables are set"""
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not API_KEY:
        missing.append("HF_TOKEN or OPENAI_API_KEY")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    
    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}", flush=True)
        print("[ERROR] Please set: API_BASE_URL, MODEL_NAME, HF_TOKEN", flush=True)
        return False
    return True

BENCHMARK = "medical_triage"
MAX_STEPS = 50


def rule_based_agent(observation):
    """
    Simple rule-based triage agent
    Follows ESI guidelines: critical first, then by acuity
    """
    if not observation.waiting_patients:
        return None
    
    # Find critical patients (ESI 1-2)
    critical_patients = []
    for p in observation.waiting_patients:
        if p.is_critical:
            critical_patients.append(p)
        elif p.chief_complaint.value in ["chest_pain", "stroke_symptoms", "head_injury"]:
            critical_patients.append(p)
    
    if critical_patients:
        # Take the most critical patient
        patient = critical_patients[0]
        
        # Assign ESI based on severity
        if patient.chief_complaint.value in ["unresponsive", "severe_bleeding"]:
            esi_level = 1
        else:
            esi_level = 2
    else:
        # Take first waiting patient, assign ESI 3
        patient = observation.waiting_patients[0]
        esi_level = 3
    
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


def llm_agent(client, observation, step_num, conversation_history):
    """
    LLM-based triage agent using OpenAI Client
    Makes triage decisions via LLM API calls
    """
    if not observation.waiting_patients:
        return None
    
    try:
        patient = observation.waiting_patients[0]
        
        # Build prompt for triage decision
        prompt = textwrap.dedent(f"""
        You are an emergency department triage expert using ESI (Emergency Severity Index) protocol.
        
        Current patient waiting:
        - ID: {patient.id}
        - Age: {patient.age}
        - Chief Complaint: {patient.chief_complaint.value.replace('_', ' ')}
        - Vital Signs: HR={patient.vital_signs.get('heart_rate', 'N/A'):.0f}, BP={patient.vital_signs.get('bp_systolic', 'N/A'):.0f}/{patient.vital_signs.get('bp_diastolic', 'N/A'):.0f}
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
            timeout=10
        )
        
        esi_str = response.choices[0].message.content.strip()
        esi_level = int(esi_str) if esi_str.isdigit() else 3
        esi_level = max(1, min(5, esi_level))
        
    except Exception as e:
        # Fallback to rule-based on error
        return rule_based_agent(observation)
    
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


def run_episode(env, episode_num, client=None, use_llm=False):
    """Run a single episode with LLM or rule-based agent"""
    observation = env.reset()
    total_reward = 0.0
    step_records = []
    step_num = 0
    conversation_history = []
    
    for step in range(MAX_STEPS):
        # Get action from LLM or rule-based agent
        if use_llm and client and HAS_OPENAI:
            action = llm_agent(client, observation, step, conversation_history)
        else:
            action = rule_based_agent(observation)
        
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
    # Validate required environment variables
    env_valid = validate_env_vars()
    
    if not env_valid:
        # Print minimal output to avoid breaking downstream parsing
        print(f"[START] task=medical_triage_inference env={BENCHMARK} model=undefined", flush=True)
        print(f"[END] success=false steps=0 score=0.0 rewards=", flush=True)
        return 1
    
    print(f"[START] task=medical_triage_inference env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    env = MedicalTriageEnv(max_steps=MAX_STEPS, random_seed=42)
    
    # Initialize OpenAI client for LLM calls
    client = None
    use_llm = False
    
    if HAS_OPENAI and API_KEY and API_BASE_URL and MODEL_NAME:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
            use_llm = True
        except Exception as e:
            print(f"[DEBUG] Failed to initialize OpenAI client: {e}", flush=True)
            use_llm = False
    
    results = []
    all_step_records = []
    all_rewards = []
    success = False
    total_steps = 0
    final_score = 0.0
    rewards_str = ""
    
    try:
        for episode in range(1, 4):
            result = run_episode(env, episode, client=client, use_llm=use_llm)
            results.append(result)
            
            # Collect step records and rewards
            for record in result["step_records"]:
                all_step_records.append(record)
                all_rewards.append(record["reward"])
                
                # Print [STEP] line in correct format
                error_val = record["error"] if record["error"] else "null"
                done_val = str(record["done"]).lower()
                print(
                    f"[STEP] step={record['step']} action={record['action']} "
                    f"reward={record['reward']:.2f} done={done_val} error={error_val}",
                    flush=True
                )
        
        # Calculate final results
        total_reward = sum(r["total_reward"] for r in results)
        total_steps = sum(len(r["step_records"]) for r in results)
        total_arrivals = sum(r["arrivals"] for r in results)
        total_lwbs = sum(r["lwbs"] for r in results)
        total_mortality = sum(r["mortality"] for r in results)
        
        # Calculate normalized score (0-1)
        # Base score on average reward per step, normalized
        avg_reward_per_step = total_reward / total_steps if total_steps > 0 else 0.0
        final_score = min(1.0, max(0.0, avg_reward_per_step))
        
        # Determine success (score >= some threshold)
        success = final_score >= 0.4
        
        # Format rewards list
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        
    finally:
        # Always emit [END] line
        success_val = str(success).lower()
        print(
            f"[END] success={success_val} steps={total_steps} score={final_score:.3f} rewards={rewards_str}",
            flush=True
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())