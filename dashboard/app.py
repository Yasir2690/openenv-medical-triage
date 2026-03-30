def create_dashboard():
    dashboard = TriageDashboard()
    
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
            waiting, resources = dashboard.get_data()
            waiting_df = pd.DataFrame(waiting, columns=["ID", "Age", "Complaint", "Wait"]) if waiting else pd.DataFrame()
            resources_df = pd.DataFrame(list(resources.items()), columns=["Metric", "Value"])
            return waiting_df, resources_df
        
        def reset_and_update():
            dashboard.reset()
            return update()
        
        def step_and_update():
            dashboard.step()
            return update()
        
        demo.load(fn=update, outputs=[patient_table, resource_table])
        reset_btn.click(fn=reset_and_update, outputs=[patient_table, resource_table])
        step_btn.click(fn=step_and_update, outputs=[patient_table, resource_table])
    
    return demo