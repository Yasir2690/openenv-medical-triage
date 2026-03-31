"""
OpenEnv Server Entry Point
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv

def main():
    """OpenEnv server entry point - called by openenv-server command"""
    print("Starting Medical Triage Environment Server...")
    env = MedicalTriageEnv()
    print("OpenEnv Medical Triage Environment ready")
    print("Environment initialized with 3 tasks: easy, medium, hard")
    return env

if __name__ == "__main__":
    main()