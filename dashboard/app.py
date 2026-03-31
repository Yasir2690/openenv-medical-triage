"""
Medical Triage Environment - Gradio Dashboard
"""

import gradio as gr
import pandas as pd
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
from src.models import TriageAction

class TriageDashboard:
    def __init__(self):
        self.env = MedicalTriageEnv(max_steps=50, random_seed=42)
        self.observation = None
        self.reset()
    
    def reset(self):
        self.observation = self.env.reset()
        return self.get_data()
    
    def step(self):
        if self.observation and self.observation.waiting_patients:
            patient = random.choice(self.observation.waiting_patients)
            action = TriageAction(
                patient_id=patient.id,
                esi_level=random.choice([1, 2, 3, 4, 5]),
                assigned_room=random.choice(self.observation.available_rooms) if self.observation.available_rooms else None,
                assigned_doctor_id=random.choice(list(self.observation.available_doctors.keys())) if self.observation.available_doctors else None,
                order_tests=[],
                initiate_resuscitation=False
            )
            self.observation, reward, done, info = self.env.step(action)
        return self.get_data()
    
    def get_data(self):
        if not self.observation:
            return [], {}
        
        waiting = []
        for p in self.observation.waiting_patients[:5]:
            waiting.append([
                p.id[:8], 
                p.age, 
                p.chief_complaint.value.replace('_', ' ').title()[:20], 
                f"{p.wait_time_minutes:.0f}min" if p.wait_time_minutes else "N/A"
            ])
        
        resources = {
            "Total Patients": self.observation.total_patients,
            "Waiting": len(self.observation.waiting_patients),
            "Triaged": len(self.observation.triaged_patients),
            "Active": len(self.observation.active_patients),
            "LWBS Rate": f"{self.observation.lwbs_rate:.1%}",
            "Rooms Available": len(self.observation.available_rooms),
            "Doctors Available": len(self.observation.available_doctors)
        }
        return waiting, resources


# Create the dashboard
dashboard_instance = TriageDashboard()

# Create Gradio interface
demo = gr.Blocks(title="Medical Triage Environment")

with demo:
    gr.Markdown("# 🏥 Medical Triage Environment")
    gr.Markdown("### AI Agent Training for Emergency Department Triage")
    gr.Markdown("This dashboard simulates an AI agent managing patient triage. Click 'Take Random Action' to see the agent make decisions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            patient_table = gr.Dataframe(
                headers=["ID", "Age", "Complaint", "Wait"], 
                label="📋 Waiting Patients",
                interactive=False
            )
        with gr.Column(scale=1):
            resource_table = gr.Dataframe(
                headers=["Metric", "Value"], 
                label="📊 Current Metrics",
                interactive=False
            )
    
    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Environment", variant="secondary", size="lg")
        step_btn = gr.Button("🎲 Take Random Action", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("### 📈 Performance Notes")
    gr.Markdown("- **ESI Level 1-2**: Critical patients who need immediate attention")
    gr.Markdown("- **LWBS**: Left Without Being Seen - patients who left due to long waits")
    gr.Markdown("- **Goal**: Minimize wait times and LWBS rate")
    
    def update():
        waiting, resources = dashboard_instance.get_data()
        waiting_df = pd.DataFrame(waiting, columns=["ID", "Age", "Complaint", "Wait"]) if waiting else pd.DataFrame()
        resources_df = pd.DataFrame(list(resources.items()), columns=["Metric", "Value"])
        return waiting_df, resources_df
    
    def reset_and_update():
        dashboard_instance.reset()
        return update()
    
    def step_and_update():
        dashboard_instance.step()
        return update()
    
    demo.load(fn=update, outputs=[patient_table, resource_table])
    reset_btn.click(fn=reset_and_update, outputs=[patient_table, resource_table])
    step_btn.click(fn=step_and_update, outputs=[patient_table, resource_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)