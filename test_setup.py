"""Quick test to verify setup"""
import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    import pandas as pd
    import transformers
    print("✓ Core packages installed")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    
print("\nNext steps:")
print("1. Add API keys to .env")
print("2. Run pilot study")
print("3. Analyze preliminary results")
