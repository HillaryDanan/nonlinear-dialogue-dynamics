#!/bin/bash

# Setup script for comprehensive pilot study

echo "======================================"
echo "NONLINEAR DIALOGUE DYNAMICS"
echo "Comprehensive Pilot Setup"
echo "======================================"

# Create all necessary files
echo "Creating files..."
touch fix_openai_key.py
touch data_collector_all_models.py
touch run_comprehensive_pilot.py
touch analyze_comprehensive_pilot.py

# Make executable
chmod +x run_comprehensive_pilot.py

echo "✓ Files created"

# Check Python version
echo ""
echo "Python version:"
python3 --version

# Install any missing packages
echo ""
echo "Checking required packages..."
python3 -c "
import sys
packages = {
    'numpy': 'numpy',
    'pandas': 'pandas', 
    'scipy': 'scipy',
    'sentence_transformers': 'sentence-transformers',
    'openai': 'openai',
    'anthropic': 'anthropic',
    'google.generativeai': 'google-generativeai',
    'dotenv': 'python-dotenv',
    'pingouin': 'pingouin'
}

missing = []
for import_name, pip_name in packages.items():
    try:
        __import__(import_name.split('.')[0])
        print(f'✓ {import_name}')
    except ImportError:
        print(f'✗ {import_name} missing')
        missing.append(pip_name)

if missing:
    print(f'\nInstall missing packages with:')
    print(f'python3 -m pip install {\" \".join(missing)}')
"

echo ""
echo "======================================"
echo "NEXT STEPS:"
echo "======================================"
echo "1. Check API keys:     python3 fix_openai_key.py"
echo "2. Test all providers: python3 run_comprehensive_pilot.py test"
echo "3. Run pilot (n=3):    python3 run_comprehensive_pilot.py"
echo "4. Analyze results:    python3 analyze_comprehensive_pilot.py"
echo ""
echo "For custom n:          python3 run_comprehensive_pilot.py 5"
echo "======================================"