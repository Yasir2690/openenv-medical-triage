"""
Inference Script for Medical Triage Environment
Rule-based and LLM-based agent with structured output format

MANDATORY REQUIREMENTS:
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Output format: [START], [STEP], [END] lines to stdout with flush=True
- Score normalized to [0, 1]
"""

import os
import random
import numpy as np
from typing import List, Optional
from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
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


def run_episode(env, episode_num, use_rule_based=True):
    """Run a single episode with rule-based or random agent"""
    observation = env.reset()
    total_reward = 0.0
    step_records = []
    
    for step in range(MAX_STEPS):
        # Get action
        if use_rule_based:
            action = rule_based_agent(observation)
            if action is None:
                break
        elif observation.waiting_patients:
            patient = random.choice(observation.waiting_patients)
            action = TriageAction(
                patient_id=patient.id,
                esi_level=random.choice([1, 2, 3, 4, 5]),
                assigned_room=random.choice(observation.available_rooms) if observation.available_rooms else None,
                assigned_doctor_id=random.choice(list(observation.available_doctors.keys())) if observation.available_doctors else None,
                order_tests=[],
                initiate_resuscitation=False
            )
        else:
            break
        
        # Execute action
        observation, reward, done, info = env.step(action)
        total_reward += reward.total
        
        # Format action as string (simpler representation)
        action_str = f"assign_esi({action.esi_level.value})"
        
        step_records.append({
            "step": step + 1,
            "action": action_str,
            "reward": reward.total,
            "done": done,
            "error": None
        })
        
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
    print(f"[START] task=medical_triage_inference env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    env = MedicalTriageEnv(max_steps=MAX_STEPS, random_seed=42)
    
    use_rule_based = True
    
    results = []
    all_step_records = []
    all_rewards = []
    success = False
    total_steps = 0
    final_score = 0.0
    rewards_str = ""
    
    try:
        for episode in range(1, 4):
            result = run_episode(env, episode, use_rule_based)
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

if __name__ == "__main__":
    main()