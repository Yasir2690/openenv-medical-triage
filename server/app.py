"""
OpenEnv Server Entry Point
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import MedicalTriageEnv
from src.models import TriageAction, TriageObservation, TriageReward

def main():
    """OpenEnv server entry point"""
    env = MedicalTriageEnv()
    print("OpenEnv Medical Triage Environment ready")
    print("Environment initialized with 3 tasks: easy, medium, hard")
    return env

# Export for OpenEnv
__all__ = ['MedicalTriageEnv', 'TriageAction', 'TriageObservation', 'TriageReward']

if __name__ == "__main__":
    main()