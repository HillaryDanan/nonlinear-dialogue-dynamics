#!/usr/bin/env python3
"""
Working pilot runner - uses what actually works
"""

import asyncio
import sys
from pathlib import Path
from typing import List
import random

# Fix path
sys.path.insert(0, str(Path(__file__).parent))

from data_collector_working import ConversationManager

class ExperimentRunner:
    """Simplified experiment runner"""
    
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.data_path = Path("data/pilot")
        self.data_path.mkdir(exist_ok=True, parents=True)
        
        # Counterbalanced conditions (Latin square)
        orders = [
            ["linear", "referenced", "hybrid"],
            ["referenced", "hybrid", "linear"],
            ["hybrid", "linear", "referenced"]
        ]
        self.conditions = orders[hash(participant_id) % 3]
        
        # Topics for multi-topic discussion
        self.topics = [
            "How might quantum computing change cryptography?",
            "What are effective strategies for sustainable urban planning?",
            "How do cognitive biases affect decision making?"
        ]
    
    async def run_task_multitopic(self, manager: ConversationManager):
        """Multi-topic discussion with topic returns"""
        
        print(f"\n  Task 1: Multi-topic Discussion")
        responses = []
        
        # First pass through topics
        for i, topic in enumerate(self.topics, 1):
            print(f"    Topic {i}/3: {topic[:50]}...")
            response = await manager.send_message(topic)
            responses.append(response)
            
        # Return to each topic
        follow_ups = [
            "Can you elaborate on the technical aspects?",
            "What about the environmental impacts?",
            "How does this apply to group decisions?"
        ]
        
        for i, follow_up in enumerate(follow_ups):
            # Reference the original topic discussion
            ref_id = i * 2  # IDs of original topic messages
            
            if manager.condition == "referenced":
                await manager.send_message(follow_up, reference_id=ref_id)
            else:
                await manager.send_message(follow_up)
                
        print(f"    ✓ Multi-topic task complete")
    
    async def run_task_clarification(self, manager: ConversationManager):
        """Clarification seeking task"""
        
        print(f"\n  Task 2: Clarification Seeking")
        
        # Complex initial statement
        initial = ("Explain the relationship between transformer architectures "
                  "and human cognitive attention, considering mathematical similarities.")
        
        response1 = await manager.send_message(initial)
        
        # Ask for clarification
        clarification = "What do you mean by 'mathematical similarities'?"
        
        if manager.condition == "referenced":
            await manager.send_message(clarification, reference_id=response1.id - 1)
        else:
            await manager.send_message(clarification)
            
        print(f"    ✓ Clarification task complete")
    
    async def run_task_correction(self, manager: ConversationManager):
        """Error correction task"""
        
        print(f"\n  Task 3: Error Correction")
        
        # Statement with error
        statement = "The speed of light is 186,000 kilometers per second."
        response1 = await manager.send_message(statement)
        
        # Correction
        correction = "Actually, it's 186,000 miles per second, or 300,000 km/s."
        
        if manager.condition == "referenced":
            await manager.send_message(correction, reference_id=response1.id - 1)
        else:
            await manager.send_message(correction)
            
        print(f"    ✓ Correction task complete")
    
    async def run_experiment(self, model_type: str = "gemini-2.5-flash"):
        """Run complete experiment for participant"""
        
        print(f"\nParticipant: {self.participant_id}")
        print(f"Model: {model_type}")
        print(f"Conditions order: {self.conditions}")
        
        for condition in self.conditions:
            print(f"\n{'='*40}")
            print(f"Condition: {condition.upper()}")
            
            manager = ConversationManager(
                self.participant_id,
                condition,
                model_type
            )
            
            # Run all tasks
            await self.run_task_multitopic(manager)
            await self.run_task_clarification(manager)
            await self.run_task_correction(manager)
            
            # Save data
            filepath = manager.save_conversation(self.data_path)
            
            print(f"\n  Summary:")
            print(f"    Messages: {len(manager.history)}")
            print(f"    References: {sum(1 for m in manager.history if m.reference_id)}")
            
            # Brief pause
            await asyncio.sleep(1)
        
        print(f"\n{'='*40}")
        print(f"✓ Participant {self.participant_id} complete!")

async def run_pilot(model_type: str = None):
    """Run pilot study with n=3"""
    
    print("\n" + "="*50)
    print("PILOT STUDY - NONLINEAR DIALOGUE DYNAMICS")
    print("="*50)
    
    # Default to cheapest working model
    if model_type is None:
        model_type = "gemini-2.5-flash"
    
    print(f"\nModel: {model_type}")
    print(f"Participants: n=3")
    print(f"Estimated cost: <$0.10")
    
    confirm = input("\nProceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return
    
    # Run 3 pilot participants
    for i in range(1, 4):
        pid = f"pilot_{i:03d}"
        runner = ExperimentRunner(pid)
        
        try:
            await runner.run_experiment(model_type)
        except Exception as e:
            print(f"\nError with {pid}: {e}")
            print("Continuing with next participant...")
            continue
    
    print("\n" + "="*50)
    print("PILOT COMPLETE!")
    print(f"Data saved in: data/pilot/")
    print("Next step: python3 analyze_pilot.py")

async def test_single():
    """Test single conversation"""
    
    print("\nTEST MODE - Single Conversation")
    print("="*40)
    
    manager = ConversationManager(
        participant_id="test",
        condition="referenced",
        model_type="gemini-2.5-flash"
    )
    
    # Test basic
    r1 = await manager.send_message("What is quantum entanglement?")
    print(f"Response 1: {r1.content[:100]}...")
    
    # Test reference
    r2 = await manager.send_message(
        "Can you explain that more simply?",
        reference_id=r1.id - 1
    )
    print(f"Response 2 (referenced): {r2.content[:100]}...")
    
    print("\n✓ Test complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            asyncio.run(test_single())
        else:
            # Run with specified model
            asyncio.run(run_pilot(model_type=sys.argv[1]))
    else:
        # Default run
        asyncio.run(run_pilot())