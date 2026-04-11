"""
OpenEnv Server Entry Point
"""

import os
import random
import sys
from datetime import datetime

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
from src.models import TriageAction


def serialize_observation(obs):
    """Convert observation to JSON-serializable dict"""
    obs_dict = obs.dict()
    if "timestamp" in obs_dict and isinstance(obs_dict["timestamp"], datetime):
        obs_dict["timestamp"] = obs_dict["timestamp"].isoformat()

    for patient_list in ["waiting_patients", "triaged_patients", "active_patients"]:
        if patient_list in obs_dict:
            for patient in obs_dict[patient_list]:
                if "arrival_time" in patient and isinstance(patient["arrival_time"], datetime):
                    patient["arrival_time"] = patient["arrival_time"].isoformat()
                if "triage_time" in patient and patient["triage_time"]:
                    patient["triage_time"] = (
                        patient["triage_time"].isoformat()
                        if isinstance(patient["triage_time"], datetime)
                        else patient["triage_time"]
                    )
                if "seen_time" in patient and patient["seen_time"]:
                    patient["seen_time"] = (
                        patient["seen_time"].isoformat()
                        if isinstance(patient["seen_time"], datetime)
                        else patient["seen_time"]
                    )
                if "discharged_time" in patient and patient["discharged_time"]:
                    patient["discharged_time"] = (
                        patient["discharged_time"].isoformat()
                        if isinstance(patient["discharged_time"], datetime)
                        else patient["discharged_time"]
                    )

    return obs_dict


def serialize_info(info):
    """Convert info dict to JSON-serializable format"""
    if not isinstance(info, dict):
        return info
    
    serialized = {}
    for key, value in info.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, dict):
            serialized[key] = serialize_info(value)
        elif isinstance(value, list):
            serialized[key] = [
                item.isoformat() if isinstance(item, datetime) else item
                for item in value
            ]
        else:
            serialized[key] = value
    
    return serialized


env = MedicalTriageEnv(max_steps=50, random_seed=42)
observation = env.reset()
print("Environment initialized")

app = FastAPI(title="Medical Triage Environment")


@app.post("/reset")
async def reset_endpoint():
    """OpenEnv reset endpoint"""
    try:
        global observation
        print("Reset endpoint called")
        observation = env.reset()
        print("Reset successful")
        obs_dict = serialize_observation(observation)
        return JSONResponse(content={"status": "ok", "observation": obs_dict})
    except Exception as e:
        print(f"Reset error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/step")
async def step_endpoint(request: Request):
    """OpenEnv step endpoint"""
    global observation
    try:
        body = await request.json()
        action_data = body.get("action", body)

        action = TriageAction(
            patient_id=action_data.get("patient_id", ""),
            esi_level=action_data.get("esi_level"),
            assigned_room=action_data.get("assigned_room"),
            assigned_doctor_id=action_data.get("assigned_doctor_id"),
            order_tests=action_data.get("order_tests", []),
            initiate_resuscitation=action_data.get("initiate_resuscitation", False),
        )
        observation, reward, done, info = env.step(action)
        obs_dict = serialize_observation(observation)
        info_dict = serialize_info(info)
        return JSONResponse(
            content={
                "observation": obs_dict,
                "reward": reward.total,
                "done": done,
                "info": info_dict,
            }
        )
    except Exception as e:
        print(f"Step error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/state")
async def state_endpoint():
    """OpenEnv state endpoint"""
    try:
        state = env.state()
        state_dict = serialize_info(state)
        return JSONResponse(content=state_dict)
    except Exception as e:
        print(f"State error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health_endpoint():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy"})


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
            initiate_resuscitation=False,
        )
        observation, reward, done, info = env.step(action)
    return get_data()


def get_data():
    waiting = []
    for p in observation.waiting_patients[:5]:
        waiting.append(
            [
                p.id[:8],
                p.age,
                p.chief_complaint.value.replace("_", " ").title()[:20],
                f"{p.wait_time_minutes:.0f}min" if p.wait_time_minutes else "N/A",
            ]
        )

    metrics = {
        "Total Patients": observation.total_patients,
        "Waiting": len(observation.waiting_patients),
        "Triaged": len(observation.triaged_patients),
        "Active": len(observation.active_patients),
        "LWBS Rate": f"{observation.lwbs_rate:.1%}",
        "Rooms Available": len(observation.available_rooms),
        "Doctors Available": len(observation.available_doctors),
    }

    waiting_df = pd.DataFrame(waiting, columns=["ID", "Age", "Complaint", "Wait"]) if waiting else pd.DataFrame()
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    return waiting_df, metrics_df


with gr.Blocks(title="Medical Triage Environment") as demo:
    gr.Markdown("# 🏥 Medical Triage Environment")
    gr.Markdown("### AI Agent Training for Emergency Department Triage")

    with gr.Row():
        patient_table = gr.Dataframe(label="📋 Waiting Patients")
        metrics_table = gr.Dataframe(label="📊 Current Metrics")

    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
        step_btn = gr.Button("🎲 Take Random Action", variant="primary")

    demo.load(fn=get_data, outputs=[patient_table, metrics_table])
    reset_btn.click(fn=reset_env, outputs=[patient_table, metrics_table])
    step_btn.click(fn=take_action, outputs=[patient_table, metrics_table])


app = gr.mount_gradio_app(app, demo, path="/")


def main():
    """Callable entry point expected by validators and script runners."""
    start_server()


def start_server():
    """OpenEnv server entry point"""
    print("Starting Medical Triage Environment server...")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()