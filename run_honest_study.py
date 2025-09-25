#!/usr/bin/env python3
"""
Main entry point for honest study execution
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from experiments.honest_experiment import HonestExperiment, ExperimentConfig

def main():
    print("=" * 60)
    print("NONLINEAR DIALOGUE DYNAMICS - HONEST SCIENCE EDITION")
    print("=" * 60)
    
    # Load config
    config = ExperimentConfig()
    
    print("\nStudy Parameters (Pre-Registered):")
    print(f"Expected effect size: d={config.expected_coherence_d}")
    print(f"Required sample size: n={config.main_n}")
    print(f"Corrected alpha: α={config.alpha:.5f}")
    print(f"Prompt length confounded: {config.prompt_length_confounded}")
    print(f"Attention weights available: {config.attention_weights_available}")
    
    # Check costs
    estimated_calls = config.main_n * 3 * 10  # participants × conditions × turns
    print(f"\nEstimated API calls: {estimated_calls}")
    print(f"Estimated cost: ${estimated_calls * 0.03:.2f}")  # Rough estimate
    
    response = input("\nProceed with honest science? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Good choice - think it through first!")
        sys.exit(0)
        
    # Initialize experiment
    experiment = HonestExperiment(config)
    
    # Show power
    powers = experiment.calculate_statistical_power()
    print("\nStatistical Power:")
    for effect, power in powers.items():
        print(f"  {effect}: {power:.2f}")
    
    print("\nStarting pilot study...")
    # Run pilot
    # experiment.run_pilot()
    
    print("\nPilot complete! Check results before proceeding to main study.")

if __name__ == "__main__":
    main()
