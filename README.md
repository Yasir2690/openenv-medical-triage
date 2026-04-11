---
title: Medical Triage Environment
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🏥 Medical Triage Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/huggingface/openenv)
[![Healthcare](https://img.shields.io/badge/Domain-Healthcare-green)](https://github.com)
[![RL](https://img.shields.io/badge/RL-Environment-orange)](https://github.com)

## Problem Statement

Emergency departments worldwide face critical challenges: long wait times, patients leaving without being seen (LWBS), and resource constraints. This environment trains AI agents to optimise triage decisions, potentially saving lives and improving healthcare delivery.

## Real-World Impact

Optimizing ED triage can deliver measurable improvements:

- **LWBS Reduction**: Current baseline ~5-10% LWBS rate; optimal triage can reduce to <2%
- **Patient Safety**: Accurate ESI assignment reduces mortality risk in critical patients by prioritizing care
- **Resource Efficiency**: Smart resource allocation improves doctor/bed utilization by 15-25%
- **Throughput**: Better triage decisions enable EDs to handle 20-30% more patients safely
- **Clinical Outcome**: Reduced wait times for critical patients (ESI 1-2) by 40%+ directly correlates with better outcomes

## Key Features

- **Real-World Simulation**: Models actual ED operations with patient acuity, deterioration, and resource constraints
- **Clinical Guidelines**: Implements ESI (Emergency Severity Index) v4 triage protocol
- **Partial Progress Rewards**: Dense reward signals for learning complex behaviours
- **3 Progressive Tasks**: Easy (basic triage) → Medium (resource allocation) → Hard (mass casualty)

## Quick Start

```bash
pip install -r requirements.txt
python inference.py
```

## Expected Performance

| Task                | Random Agent | Rule-Based | Target |
| ------------------- | ------------ | ---------- | ------ |
| Basic Triage        | 0.35         | 0.55       | 0.70   |
| Resource Allocation | 0.28         | 0.48       | 0.60   |
| Mass Casualty       | 0.22         | 0.40       | 0.50   |

## Environment Specification

### Action Space

| Field                    | Type        | Description                                  |
| ------------------------ | ----------- | -------------------------------------------- |
| `esi_level`              | int 1–5     | ESI priority (1=resuscitation, 5=non-urgent) |
| `assigned_room`          | str \| None | Room ID from available rooms                 |
| `assigned_doctor_id`     | str \| None | Doctor ID from available doctors             |
| `order_tests`            | list[str]   | Diagnostic tests to order                    |
| `initiate_resuscitation` | bool        | Activate code blue protocol                  |

### Observation Space

- Waiting, triaged, and active patients with full clinical data
- Available resources (rooms, doctors, equipment)
- Current wait times by ESI level
- LWBS rate and performance metrics

### Reward Function

| Component            | Range    | Criteria                                   |
| -------------------- | -------- | ------------------------------------------ |
| Patient outcome      | 0 – 0.5  | Correct ESI assignment, reduced wait times |
| Wait time reduction  | 0 – 0.3  | Prioritising critical patients             |
| Resource utilisation | 0 – 0.2  | Efficient room/doctor use                  |
| Penalties            | -0.5 – 0 | LWBS events, mortality, invalid actions    |

## Project Structure

```
openenv-medical-triage/
├── inference.py          # LLM + rule-based baseline agent with grader scoring
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Container configuration
├── requirements.txt      # Dependencies
├── test_env.py           # Smoke tests
└── src/
    ├── environment.py    # Main environment (step/reset/state)
    ├── models.py         # Pydantic models
    ├── triage_logic.py   # ESI clinical decision support
    ├── simulation.py     # Patient generation
    ├── graders.py        # 3 task graders (easy/medium/hard)
    └── reward.py         # Reward schema documentation
```

## Setup Instructions

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke tests
python test_env.py

# Run baseline agent (prints per-task grader scores)
python inference.py
```

### Docker

```bash
docker build -t medical-triage .
docker run -p 7860:7860 medical-triage
```

### With LLM Agent

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
python inference.py
```

### Required Environment Configuration (Submission)

The submission runner expects these variables to be defined:

- `API_BASE_URL`: The API endpoint for the LLM.
- `MODEL_NAME`: The model identifier used for inference.
- `HF_TOKEN`: Hugging Face token / API key.

PowerShell example:

```powershell
$env:API_BASE_URL="https://api-inference.huggingface.co/v1"
$env:MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
$env:HF_TOKEN="your_token_here"
python inference.py
```

### Runtime + Resource Limits (2 vCPU / 8 GB)

The inference script includes runtime guards so execution can finish within 20 minutes on limited machines.

Optional tuning environment variables:

- `MAX_RUNTIME_SECONDS` (default: `1080`) - Global wall-clock cap for all tasks.
- `MAX_TASK_RUNTIME_SECONDS` (default: `360`) - Per-task wall-clock cap.
- `MAX_LLM_CALLS_PER_TASK` (default: `45`) - Upper bound on API calls per task.
- `LLM_EVERY_N_STEPS` (default: `3`) - Call LLM every N steps; heuristic actions in between.
- `LLM_TIMEOUT_SECONDS` (default: `12`) - Network timeout per LLM call.

Suggested PowerShell settings for strict <20 min runtime:

```powershell
$env:MAX_RUNTIME_SECONDS="1080"
$env:MAX_TASK_RUNTIME_SECONDS="360"
$env:MAX_LLM_CALLS_PER_TASK="45"
$env:LLM_EVERY_N_STEPS="3"
$env:LLM_TIMEOUT_SECONDS="12"
python inference.py
```

## Baseline Scores (Rule-Based Agent)

```
FINAL GRADER SCORES (0.0 – 1.0)
  ✓  easy      score=0.55  target=0.70
  ✓  medium    score=0.48  target=0.60
  ✓  hard      score=0.40  target=0.50
```

## License

MIT

**Team:** Open Source Minds  
**Contact:** yasirasfaque91@gmail.com
