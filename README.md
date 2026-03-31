---
title: Medical Triage Environment
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

#  Medical Triage Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/huggingface/openenv)
[![Healthcare](https://img.shields.io/badge/Domain-Healthcare-green)](https://github.com)
[![RL](https://img.shields.io/badge/RL-Environment-orange)](https://github.com)

# Problem Statement
Emergency departments worldwide face critical challenges: long wait times, patient LWBS (Left Without Being Seen), and resource constraints. This environment trains AI agents to optimize triage decisions, potentially saving lives and improving healthcare delivery.

# Key Features
- **Real-World Simulation**: Models actual ED operations with patient acuity, deterioration, and resource constraints
- **Clinical Guidelines**: Implements ESI (Emergency Severity Index) v4 triage protocol
- **Partial Progress Rewards**: Dense reward signals for learning complex behaviors
- **3 Progressive Tasks**: Easy (basic triage) → Medium (resource allocation) → Hard (mass casualty)

# Quick Start
```bash
pip install -r requirements.txt
python inference.py

 Performance
Task	Random Agent	Target
Basic Triage	0.45	0.70
Resource Allocation	0.38	0.60
Mass Casualty	0.32	0.50

 Environment Specification
Action Space
esi_level: Assign ESI 1-5 (1=critical, 5=non-urgent)

assigned_room: Assign to available room

assigned_doctor_id: Assign to available doctor

order_tests: Order diagnostic tests

initiate_resuscitation: Activate code blue

Observation Space
Waiting, triaged, and active patients with clinical data

Available resources (rooms, doctors, equipment)

Current wait times by ESI level

LWBS rate and performance metrics

Reward Function
Patient outcomes: 0-0.5 (correct ESI assignment, reduced wait times)

Wait time reduction: 0-0.3 (prioritizing critical patients)

Resource utilization: 0-0.2 (efficient resource use)

Penalties: LWBS, mortality, inefficient actions

Why This Wins
Real-World Impact: Direct application to healthcare, Meta's focus area
Technical Excellence: Full OpenEnv compliance, typed models, Dockerized
Clinical Accuracy: Based on ESI guidelines used in 70% of US EDs
Scalability: Can be extended to more clinical scenarios

Project Structure

openenv-medical-triage/
├── inference.py          # Baseline agent
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Container configuration
├── requirements.txt      # Dependencies
├── src/
│   ├── environment.py    # Main environment
│   ├── models.py         # Pydantic models
│   ├── triage_logic.py   # Clinical decision support
│   ├── simulation.py     # Patient generation
│   ├── graders.py        # 3 task graders
│   └── reward.py         # Reward calculation
└── dashboard/
    └── app.py            # Gradio dashboard


# Setup Instructions

## Local Development
```bash
# Clone repository
git clone https://github.com/yasir2690/medical_triage_env
cd medical_triage_env

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_env.py

# Run baseline agent
python inference.py

Docker
docker build -t medical-triage .
docker run -p 7860:7860 medical-triage

Baseline Scores (Random Agent)
Episode 1: Reward 6.52, 12 patients, 0% LWBS, 0% mortality

Episode 2: Reward 3.24, 6 patients, 0% LWBS, 0% mortality

Episode 3: Reward 3.25, 6 patients, 0% LWBS, 0% mortality

Average Reward: 4.34 across 3 episodes

 License
MIT

Team Name: Open Source Minds

Contact: yasirasfaque91@gmail.com