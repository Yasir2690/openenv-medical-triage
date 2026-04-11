#!/usr/bin/env python3
"""Test script to validate deterioration mechanics"""

from src.environment import MedicalTriageEnv
from src.models import TriageAction, ESILevel
import traceback

try:
    env = MedicalTriageEnv()
    obs = env.reset()
    
    # Run a few steps
    print('Testing deterioration mechanics...')
    print(f'Initial patients: {len(obs.waiting_patients)}')
    
    for step_num in range(10):
        if obs.waiting_patients:
            patient = obs.waiting_patients[0]
            action = TriageAction(
                patient_id=patient.id,
                esi_level=ESILevel.URGENT,
                assigned_room='Room1',
                assigned_doctor_id='Doc1' if obs.available_doctors else None,
                order_tests=[],
                initiate_resuscitation=False
            )
            obs, reward, done, info = env.step(action)
            
            # Check if any patients have deteriorated
            has_deteriorated = any(p.has_deteriorated for p in (obs.waiting_patients + obs.triaged_patients))
            if has_deteriorated:
                print(f'Step {step_num}: DETERIORATION DETECTED!')
                print(f'  Reward component: {float(reward):.4f}')
        
        if done:
            print(f'Episode done at step {step_num}')
            break
    
    print('✓ Deterioration mechanics test passed - no errors')
    print(f'Final metrics: {info.get("metrics", {}) if info else {}}')
    
except Exception as e:
    print(f'✗ Error: {e}')
    traceback.print_exc()
