"""
Updated model configuration for September 2025
Including new Gemini 2.5 models
"""

MODELS = {
    # Google models - NEW 2.5 versions!
    "gemini-2.5-flash": {
        "provider": "google",
        "name": "models/gemini-2.5-flash",
        "cost_per_1k_input": 0.000075,  # Assuming similar to 1.5
        "cost_per_1k_output": 0.0003,
        "description": "Latest Gemini flash - Sept 2025"
    },
    "gemini-2.5-flash-lite": {
        "provider": "google",
        "name": "models/gemini-2.5-flash-lite",
        "cost_per_1k_input": 0.00005,  # Even cheaper!
        "cost_per_1k_output": 0.0002,
        "description": "Ultra-light version for simple tasks"
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "name": "models/gemini-2.0-flash",
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
        "description": "Stable 2.0 version"
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "name": "models/gemini-1.5-flash",
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
        "description": "Older but stable flash model"
    },
    
    # Anthropic models (working despite deprecation warning)
    "claude-haiku": {
        "provider": "anthropic",
        "name": "claude-3-5-haiku-20241022",
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.004,
        "description": "Fast, cheap Claude model"
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "name": "claude-3-5-sonnet-20241022",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "description": "High quality Claude (deprecates Oct 2025)"
    },
    
    # OpenAI - will work once we fix proxy issue
    "gpt-4o-mini": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "description": "OpenAI's efficient model (if it works)"
    }
}

def estimate_experiment_cost(n_participants: int = 64, 
                            turns_per_condition: int = 10,
                            conditions: int = 3,
                            model: str = "gemini-2.5-flash"):
    """Estimate total experiment cost"""
    
    if model not in MODELS:
        return None
        
    model_info = MODELS[model]
    
    # Conservative estimates
    avg_input_tokens = 120  # Slightly higher for safety
    avg_output_tokens = 150
    
    total_turns = n_participants * turns_per_condition * conditions
    total_input_tokens = total_turns * avg_input_tokens
    total_output_tokens = total_turns * avg_output_tokens
    
    input_cost = (total_input_tokens / 1000) * model_info["cost_per_1k_input"]
    output_cost = (total_output_tokens / 1000) * model_info["cost_per_1k_output"]
    total_cost = input_cost + output_cost
    
    return total_cost

def print_cost_analysis(n_participants: int):
    """Print cost analysis for all working models"""
    print(f"\nCost Analysis for n={n_participants}")
    print("="*40)
    
    working_models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite", 
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "claude-haiku"
    ]
    
    for model in working_models:
        cost = estimate_experiment_cost(n_participants, model=model)
        if cost:
            print(f"{model:20} ${cost:.2f}")
    
    print(f"\n✓ Recommendation: Use gemini-2.5-flash-lite for pilot")
    print(f"  Only ${estimate_experiment_cost(3, model='gemini-2.5-flash-lite'):.2f} for n=3!")