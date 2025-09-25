"""
Model configuration for September 2025
Costs and models current as of this date
"""

MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "description": "Cheaper, faster than GPT-3.5-turbo with better performance"
    },
    "claude-sonnet": {
        "provider": "anthropic", 
        "name": "claude-3-5-sonnet-20241022",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "description": "High quality, current Sonnet model"
    },
    "claude-haiku": {
        "provider": "anthropic",
        "name": "claude-3-5-haiku-20241022", 
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.004,
        "description": "Fast, cheap, good for simple tasks"
    },
    "gemini-flash": {
        "provider": "google",
        "name": "gemini-1.5-flash",
        "cost_per_1k_input": 0.000075,  # Very cheap!
        "cost_per_1k_output": 0.0003,
        "description": "Extremely cheap, fast, good enough for many tasks"
    },
    "gemini-pro": {
        "provider": "google",
        "name": "gemini-1.5-pro", 
        "cost_per_1k_input": 0.0035,
        "cost_per_1k_output": 0.0105,
        "description": "More capable Gemini model"
    }
}

def estimate_experiment_cost(n_participants: int = 64, 
                            turns_per_condition: int = 10,
                            conditions: int = 3,
                            model: str = "gpt-4o-mini"):
    """Estimate total experiment cost"""
    
    if model not in MODELS:
        print(f"Unknown model: {model}")
        return
        
    model_info = MODELS[model]
    
    # Rough estimates
    avg_input_tokens = 100  # Per turn
    avg_output_tokens = 150  # Per turn
    
    total_turns = n_participants * turns_per_condition * conditions
    total_input_tokens = total_turns * avg_input_tokens
    total_output_tokens = total_turns * avg_output_tokens
    
    input_cost = (total_input_tokens / 1000) * model_info["cost_per_1k_input"]
    output_cost = (total_output_tokens / 1000) * model_info["cost_per_1k_output"]
    total_cost = input_cost + output_cost
    
    print(f"\nCost Estimate for {model}:")
    print(f"  Participants: {n_participants}")
    print(f"  Total API calls: {total_turns}")
    print(f"  Input cost: ${input_cost:.2f}")
    print(f"  Output cost: ${output_cost:.2f}")
    print(f"  TOTAL: ${total_cost:.2f}")
    
    return total_cost

def compare_all_models(n_participants: int = 64):
    """Compare costs across all models"""
    print("Cost Comparison Across Models")
    print("=" * 50)
    
    costs = {}
    for model in MODELS:
        cost = estimate_experiment_cost(n_participants, model=model)
        costs[model] = cost
        
    # Sort by cost
    sorted_costs = sorted(costs.items(), key=lambda x: x[1])
    
    print("\nRanked by cost (cheapest first):")
    for model, cost in sorted_costs:
        print(f"  {model:15} ${cost:.2f}")
        
    print(f"\nRecommendation: Use {sorted_costs[0][0]} for pilot")
    print(f"Then use {sorted_costs[1][0]} for main study if budget allows")

if __name__ == "__main__":
    # Compare all options
    compare_all_models(n_participants=15)  # Pilot
    print("\n" + "="*50)
    compare_all_models(n_participants=64)  # Main study