"""
OpenEnv Server Entry Point
"""

import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
import main

def main():
    """OpenEnv server entry point"""
    print("Starting Medical Triage Environment server...")
    uvicorn.run(main_app.app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()