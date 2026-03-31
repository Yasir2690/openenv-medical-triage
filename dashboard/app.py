"""
Medical Triage Environment - Gradio Dashboard with OpenEnv Endpoints
"""

import gradio as gr
import pandas as pd
import sys
import os
import json
import random
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
from src.models import TriageAction, TriageObservation, TriageReward

# Create FastAPI app for OpenEnv endpoints
app = FastAPI(title="Medical Triage Environment", version="1.0.0")

# Initialize environment
env = MedicalTriageEnv(max_steps=50, random_seed=42)
observation = env.reset()

# OpenEnv API Models
class ResetResponse(BaseModel):
    status: str
    observation: dict

class StepRequest(BaseModel):
    action: dict

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict

# OpenEnv Endpoints
@app.get("/reset")
async def reset_endpoint():
    """OpenEnv reset endpoint"""
    global env, observation
    observation = env.reset()
    return ResetResponse(
        status="ok",
        observation=observation.dict()
    )

@app.post("/step")
async def step_endpoint(request: StepRequest):
    """OpenEnv step endpoint"""
    global env, observation
    # Convert dict to TriageAction
    action = TriageAction(**request.action)
    observation, reward, done, info = env.step(action)
    return StepResponse(
        observation=observation.dict(),
        reward=reward.total,
        done=done,
        info=info
    )

@app.get("/state")
async def state_endpoint():
    """OpenEnv state endpoint"""
    return env.state()

# Gradio Dashboard
def reset_env():
    global observation
    observation = env.reset()
    return get_data()

def take_action():
    global observation
    if observation.waiting_patients:
        patient = random.choice(observation.waiting_patients)
        action = TriageAction(
            patient_id=patient.id,
            esi_level=random.choice([1, 2, 3, 4, 5]),
            assigned_room=random.choice(observation.available_rooms) if observation.available_rooms else None,
            assigned_doctor_id=random.choice(list(observation.available_doctors.keys())) if observation.available_doctors else None,
            order_tests=[],
            initiate_resuscitation=False
        )
        observation, reward, done, info = env.step(action)
    return get_data()

def get_data():
    waiting = []
    for p in observation.waiting_patients[:5]:
        waiting.append([
            p.id[:8], 
            p.age, 
            p.chief_complaint.value.replace('_', ' ').title()[:20],
            f"{p.wait_time_minutes:.0f}min" if p.wait_time_minutes else "N/A"
        ])
    
    metrics = {
        "Total Patients": observation.total_patients,
        "Waiting": len(observation.waiting_patients),
        "Triaged": len(observation.triaged_patients),
        "Active": len(observation.active_patients),
        "LWBS Rate": f"{observation.lwbs_rate:.1%}",
        "Rooms Available": len(observation.available_rooms),
        "Doctors Available": len(observation.available_doctors)
    }
    
    waiting_df = pd.DataFrame(waiting, columns=["ID", "Age", "Complaint", "Wait"]) if waiting else pd.DataFrame()
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    return waiting_df, metrics_df

# Create Gradio interface
with gr.Blocks(title="Medical Triage Environment") as demo:
    gr.Markdown("# 🏥 Medical Triage Environment")
    gr.Markdown("### AI Agent Training for Emergency Department Triage")
    
    with gr.Row():
        with gr.Column(scale=1):
            patient_table = gr.Dataframe(label="📋 Waiting Patients")
        with gr.Column(scale=1):
            metrics_table = gr.Dataframe(label="📊 Current Metrics")
    
    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
        step_btn = gr.Button("🎲 Take Random Action", variant="primary")
    
    demo.load(fn=get_data, outputs=[patient_table, metrics_table])
    reset_btn.click(fn=reset_env, outputs=[patient_table, metrics_table])
    step_btn.click(fn=take_action, outputs=[patient_table, metrics_table])

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)