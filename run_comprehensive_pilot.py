#!/usr/bin/env python3
"""
Comprehensive pilot study across all three model providers
Implements full factorial design: 3 conditions × 3 providers × n participants
Following experimental design principles (Montgomery, 2017)
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
import random

# Fix imports
sys.path.insert(0, str(Path(__file__).parent))

from data_collector_all_models import ConversationManager, UnifiedModelInterface

class ComprehensiveExperimentRunner:
    """
    Runs experiments across all providers systematically
    Counterbalanced for both condition and provider order effects
    """
    
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.data_path = Path("data/comprehensive_pilot")
        self.data_path.mkdir(exist_ok=True, parents=True)
        
        # Get counterbalanced orders
        self.conditions = self._get_condition_order()
        self.providers = self._get_provider_order()
        
        # Experimental tasks
        self.topics = [
            "How might quantum computing change cryptography?",
            "What are effective strategies for sustainable urban planning?",
            "How do cognitive biases affect decision making?"
        ]
    
    def _get_condition_order(self) -> List[str]:
        """Latin square for condition order (3x3)"""
        orders = [
            ["linear", "referenced", "hybrid"],
            ["referenced", "hybrid", "linear"],
            ["hybrid", "linear", "referenced"]
        ]
        return orders[hash(self.participant_id + "cond") % 3]
    
    def _get_provider_order(self) -> List[str]:
        """Latin square for provider order (3x3)"""
        orders = [
            ["openai", "anthropic", "google"],
            ["anthropic", "google", "openai"],
            ["google", "openai", "anthropic"]
        ]
        return orders[hash(self.participant_id + "prov") % 3]
    
    async def run_task_multitopic(self, manager: ConversationManager) -> Dict:
        """Multi-topic discussion with returns to earlier topics"""
        
        task_metrics = {
            "task": "multitopic",
            "start_time": datetime.now().isoformat()
        }
        
        responses = []
        
        # Initial pass through topics
        for i, topic in enumerate(self.topics):
            response = await manager.send_message(topic)
            responses.append(response)
            
        # Return to each topic with elaboration request
        follow_ups = [
            "Can you elaborate on the technical challenges?",
            "What about the environmental and social impacts?",
            "How does this apply to real-world group decisions?"
        ]
        
        for i, follow_up in enumerate(follow_ups):
            # Reference the original topic (message IDs 0, 2, 4)
            ref_id = i * 2
            
            if manager.condition == "referenced":
                await manager.send_message(follow_up, reference_id=ref_id)
            else:
                await manager.send_message(follow_up)
        
        # Cross-topic connections
        cross_connections = [
            ("How might quantum computing affect urban planning?", [0, 2]),
            ("What cognitive biases affect cryptography adoption?", [0, 4]),
            ("How do urban planning decisions reflect cognitive biases?", [2, 4])
        ]
        
        for question, ref_topics in cross_connections:
            # For referenced condition, pick one of the relevant topics
            if manager.condition == "referenced" and ref_topics:
                ref_id = ref_topics[0] * 2
                await manager.send_message(question, reference_id=ref_id)
            else:
                await manager.send_message(question)
        
        task_metrics["end_time"] = datetime.now().isoformat()
        task_metrics["n_messages"] = len(manager.history)
        
        return task_metrics
    
    async def run_task_clarification(self, manager: ConversationManager) -> Dict:
        """Clarification seeking task"""
        
        task_metrics = {
            "task": "clarification",
            "start_time": datetime.now().isoformat()
        }
        
        # Complex technical statement
        initial = """The transformer architecture's self-attention mechanism 
        exhibits mathematical parallels to human cognitive attention through 
        selective information processing and context-dependent weighting."""
        
        response1 = await manager.send_message(initial)
        
        # Series of clarifications
        clarifications = [
            "What do you mean by 'mathematical parallels'?",
            "Can you explain the selective information processing?",
            "How is the weighting context-dependent?"
        ]
        
        for clarification in clarifications:
            if manager.condition == "referenced":
                # Reference the initial statement
                await manager.send_message(clarification, reference_id=0)
            else:
                await manager.send_message(clarification)
        
        task_metrics["end_time"] = datetime.now().isoformat()
        return task_metrics
    
    async def run_task_correction(self, manager: ConversationManager) -> Dict:
        """Error correction task"""
        
        task_metrics = {
            "task": "correction",
            "start_time": datetime.now().isoformat()
        }
        
        # Statements with errors to correct
        error_pairs = [
            ("The Earth orbits the Sun in 265 days.", 
             "Sorry, I meant 365 days."),
            ("Water boils at 100 degrees Fahrenheit at sea level.",
             "Actually, that should be 100 degrees Celsius or 212 Fahrenheit."),
            ("The speed of light is 186,000 kilometers per second.",
             "Correction: it's 186,000 miles per second, or about 300,000 km/s.")
        ]
        
        for error, correction in error_pairs:
            response = await manager.send_message(error)
            
            if manager.condition == "referenced":
                await manager.send_message(correction, reference_id=response.id - 1)
            else:
                await manager.send_message(correction)
        
        task_metrics["end_time"] = datetime.now().isoformat()
        return task_metrics
    
    async def run_single_combination(self, condition: str, provider: str) -> Dict:
        """Run one condition-provider combination"""
        
        print(f"\n  Testing: {condition} × {provider}")
        
        try:
            manager = ConversationManager(
                self.participant_id,
                condition,
                provider
            )
            
            # Check if model initialized
            if not manager.model.initialized:
                print(f"    ✗ {provider} not available, skipping")
                return {
                    "status": "skipped",
                    "reason": "model_not_initialized"
                }
            
            # Run all tasks
            task_results = []
            
            print(f"    Running multitopic task...")
            task_results.append(await self.run_task_multitopic(manager))
            
            print(f"    Running clarification task...")
            task_results.append(await self.run_task_clarification(manager))
            
            print(f"    Running correction task...")
            task_results.append(await self.run_task_correction(manager))
            
            # Save conversation
            filepath = manager.save_conversation(self.data_path)
            
            # Summary
            print(f"    ✓ Complete: {len(manager.history)} messages")
            print(f"    ✓ References: {sum(1 for m in manager.history if m.reference_id)}")
            print(f"    ✓ Saved to: {filepath.name}")
            
            return {
                "status": "success",
                "filepath": str(filepath),
                "n_messages": len(manager.history),
                "task_results": task_results
            }
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_full_experiment(self):
        """Run complete 3×3 factorial experiment"""
        
        print(f"\n{'='*60}")
        print(f"PARTICIPANT: {self.participant_id}")
        print(f"Condition order: {self.conditions}")
        print(f"Provider order: {self.providers}")
        print(f"{'='*60}")
        
        results = {}
        
        # Full factorial: each condition with each provider
        for condition in self.conditions:
            results[condition] = {}
            
            print(f"\nCONDITION: {condition.upper()}")
            
            for provider in self.providers:
                result = await self.run_single_combination(condition, provider)
                results[condition][provider] = result
                
                # Brief pause between runs
                await asyncio.sleep(1)
        
        # Save experiment summary
        summary = {
            "participant_id": self.participant_id,
            "timestamp": datetime.now().isoformat(),
            "condition_order": self.conditions,
            "provider_order": self.providers,
            "results": results
        }
        
        summary_path = self.data_path / f"{self.participant_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ PARTICIPANT {self.participant_id} COMPLETE")
        print(f"✓ Summary saved to: {summary_path.name}")
        print(f"{'='*60}")

async def test_all_providers():
    """Quick test of all three providers"""
    
    print("\nTESTING ALL PROVIDERS")
    print("="*50)
    
    providers = ["openai", "anthropic", "google"]
    
    for provider in providers:
        print(f"\nTesting {provider}...")
        
        try:
            model = UnifiedModelInterface(provider)
            
            if model.initialized:
                response, tokens = await model.generate("Say 'Working!'", None)
                print(f"✓ {provider}: {response[:50]}")
            else:
                print(f"✗ {provider}: Not initialized")
                
        except Exception as e:
            print(f"✗ {provider}: {e}")
    
    print("\n" + "="*50)

async def run_comprehensive_pilot(n_participants: int = 3):
    """Run comprehensive pilot with all providers"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE PILOT STUDY")
    print("Nonlinear Dialogue Dynamics")
    print("="*60)
    
    print(f"\nDesign: 3×3 factorial")
    print(f"  Conditions: linear, referenced, hybrid")
    print(f"  Providers: OpenAI, Anthropic, Google")
    print(f"  Participants: n={n_participants}")
    
    # Estimate costs
    print(f"\nEstimated costs:")
    print(f"  Per participant: ~$0.15 (if all providers work)")
    print(f"  Total: ~${n_participants * 0.15:.2f}")
    
    confirm = input("\nProceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return
    
    # Test providers first
    await test_all_providers()
    
    proceed = input("\nContinue with pilot? (y/n): ")
    if proceed.lower() != 'y':
        print("Cancelled")
        return
    
    # Run participants
    for i in range(1, n_participants + 1):
        participant_id = f"pilot_{i:03d}"
        
        runner = ComprehensiveExperimentRunner(participant_id)
        
        try:
            await runner.run_full_experiment()
        except Exception as e:
            print(f"\nError with {participant_id}: {e}")
            continue
    
    print("\n" + "="*60)
    print("COMPREHENSIVE PILOT COMPLETE!")
    print(f"Data saved in: data/comprehensive_pilot/")
    print("Next: python3 analyze_comprehensive_pilot.py")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            asyncio.run(test_all_providers())
        elif sys.argv[1].isdigit():
            asyncio.run(run_comprehensive_pilot(int(sys.argv[1])))
        else:
            print("Usage:")
            print("  python3 run_comprehensive_pilot.py       # Run with n=3")
            print("  python3 run_comprehensive_pilot.py test  # Test providers")
            print("  python3 run_comprehensive_pilot.py 5     # Run with n=5")
    else:
        asyncio.run(run_comprehensive_pilot())