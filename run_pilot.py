#!/usr/bin/env python3
"""
Pilot study runner - Test with n=3 first to verify everything works
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from experiments.run_experiment import ExperimentRunner

async def run_pilot():
    """Run pilot with 3 participants"""
    
    print("PILOT STUDY - NONLINEAR DIALOGUE DYNAMICS")
    print("="*50)
    print("Testing with n=3 participants first")
    print("This will validate:")
    print("- API connections work")
    print("- Data collection pipeline")
    print("- Task designs elicit expected responses")
    print("- Preliminary effect size estimation")
    print()
    
    pilot_ids = ["pilot_001", "pilot_002", "pilot_003"]
    
    for pid in pilot_ids:
        print(f"\n{'='*50}")
        print(f"Running participant: {pid}")
        print(f"{'='*50}")
        
        runner = ExperimentRunner(pid)
        await runner.run_full_experiment(model_type="gpt-3.5")
        
        print(f"\n✓ Participant {pid} complete")
        
    print("\n" + "="*50)
    print("PILOT COMPLETE!")
    print("Check data/experiments/ for results")
    print("Run analysis/analyze_pilot.py next")

if __name__ == "__main__":
    asyncio.run(run_pilot())
