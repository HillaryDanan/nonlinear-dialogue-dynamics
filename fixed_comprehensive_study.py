#!/usr/bin/env python3
"""
Fixed Comprehensive Study: Non-Linear Dialogue Dynamics with Recursion Analysis
Addressing all of vibe Claude's suggestions
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StudyResults:
    """Container for all experimental results"""
    coherence_by_condition: Dict[str, List[float]]
    degradation_patterns: Dict[str, str]  # model -> pattern type
    critical_depths: Dict[str, int]  # model -> breaking point
    effect_sizes: Dict[str, float]  # condition pairs -> Cohen's d

class ComprehensiveExperiment:
    """
    Full implementation addressing:
    1. General coherence maintenance
    2. Recursion depth effects
    3. Model-specific degradation signatures
    4. Temporal inconsistency handling
    """
    
    def __init__(self):
        # Experimental conditions (your friend's suggestion)
        self.conditions = [
            "linear",  # Baseline
            "shallow_reference",  # N-1 only
            "deep_recursion",  # N-3, N-5, N-7
            "contradictory",  # Reference with contradiction
            "branching"  # Multiple simultaneous references
        ]
        
        # 50 prompts for proper power (not 6!)
        self.prompts = self.generate_prompt_battery()
        
        # Recursion depths to test
        self.depths = [1, 3, 5, 7, 9, 11]  # Test beyond Miller's 7±2
        
        self.results_path = Path("data/comprehensive_results")
        self.results_path.mkdir(exist_ok=True, parents=True)
    
    def generate_prompt_battery(self) -> List[str]:
        """Generate 50 diverse prompts for statistical power"""
        
        topics = [
            # Technical
            "quantum entanglement", "blockchain consensus", "neural plasticity",
            "CRISPR gene editing", "dark matter", "protein folding",
            
            # Abstract
            "consciousness", "free will", "emergence", "causality",
            "information theory", "complexity", "entropy",
            
            # Applied
            "climate modeling", "vaccine development", "urban planning",
            "economic inflation", "machine learning bias", "cryptography",
            
            # Interdisciplinary
            "quantum computing in cryptography", "AI in medicine",
            "psychology of economics", "physics of consciousness"
        ]
        
        questions = [
            "What are the fundamental principles?",
            "How does this relate to broader systems?",
            "What are the current limitations?",
            "What contradictions exist in our understanding?",
            "How might this evolve in the future?"
        ]
        
        prompts = []
        for topic in topics[:10]:  # 10 topics
            for question in questions:  # 5 questions each = 50 total
                prompts.append(f"{topic}: {question}")
        
        return prompts
    
    def test_recursion_depth(self, model, max_depth: int = 11) -> Dict:
        """
        Test how coherence degrades with recursion depth
        Addresses: 'Do they maintain coherence better or worse when looping?'
        """
        
        results = {
            'depths': [],
            'coherences': [],
            'response_times': []
        }
        
        # Build conversation with increasing depth
        conversation = []
        base_topic = "quantum computing"
        
        for depth in range(max_depth):
            if depth == 0:
                prompt = f"Explain {base_topic}"
            else:
                # Reference increasingly distant message
                ref_idx = max(0, len(conversation) - depth)
                ref_content = conversation[ref_idx]['content'][:50]
                prompt = f"Returning to your point about '{ref_content}'..."
            
            start = datetime.now()
            response = model.generate(prompt)
            elapsed = (datetime.now() - start).total_seconds()
            
            # Calculate coherence with referenced content
            if depth >