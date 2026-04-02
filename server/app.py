"""
OpenEnv Server Entry Point
"""

import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
import main  # Load the FastAPI app from main.py

def start_server():
    """OpenEnv server entry point"""
    print("Starting Medical Triage Environment server...")
    uvicorn.run(main.app, host="0.0.0.0", port=7860)

# This is required for the entry point to be callable
if __name__ == "__main__":
    start_server()