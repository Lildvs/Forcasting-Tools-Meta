#!/usr/bin/env python
"""
Personality Templates System - Usage Example

This script demonstrates how to use the personality templates system
to create customized prompts for forecasting.
"""

from forecasting_tools.personality_templates import (
    PersonalityConfig,
    PersonalityManager,
    TemplateManager,
)


def demonstrate_personality_traits():
    """Demonstrate how to use the personality traits system."""
    print("=" * 80)
    print("PERSONALITY TRAITS EXAMPLE")
    print("=" * 80)
    
    # Create a template manager
    template_manager = TemplateManager(bot_version="q2")
    
    # Load the default research template
    template = template_manager.get_research_template("default_research")
    print("Original template (excerpts):")
    print("\n".join(template.split("\n")[:5]))
    print("...")
    print("\n")
    
    # Apply different personality traits
    traits_configs = {
        "Cautious": {
            "reasoning_depth": "deep",
            "uncertainty_approach": "cautious",
            "expert_persona": "forecaster",
            "thinking_style": "systematic"
        },
        "Creative": {
            "reasoning_depth": "medium",
            "uncertainty_approach": "confident",
            "expert_persona": "forecaster",
            "thinking_style": "creative"
        },
        "Economist": {
            "reasoning_depth": "deep",
            "uncertainty_approach": "balanced",
            "expert_persona": "economist",
            "thinking_style": "analytical"
        }
    }
    
    for name, traits in traits_configs.items():
        print(f"{name} Personality:")
        template_with_traits = template_manager.apply_personality_traits(template, traits)
        
        # Format with a sample question
        formatted_template = template_with_traits.format(
            question="Will AI safety become a major political issue by 2025?"
        )
        
        # Print just enough to show the differences
        print("\n".join(formatted_template.split("\n")[10:15]))
        print("...")
        print("\n")


def demonstrate_personality_configs():
    """Demonstrate how to use pre-defined personality configurations."""
    print("=" * 80)
    print("PERSONALITY CONFIGURATIONS EXAMPLE")
    print("=" * 80)
    
    # List available personalities
    available_personalities = PersonalityConfig.get_available_personalities()
    print(f"Available personalities: {', '.join(available_personalities)}")
    print("\n")
    
    # Load and print details for each personality
    for personality_name in available_personalities:
        config = PersonalityConfig(personality_name=personality_name)
        print(f"{config.get_name()}: {config.get_description()}")
        print(f"  Traits: {config.get_traits_config()}")
        print(f"  Thinking: {config.get_thinking_parameters()}")
        print("\n")


def demonstrate_personality_manager():
    """Demonstrate how to use the PersonalityManager for forecasting bots."""
    print("=" * 80)
    print("PERSONALITY MANAGER EXAMPLE")
    print("=" * 80)
    
    # Create personality managers with different configurations
    managers = {
        "Balanced Q2": PersonalityManager(bot_version="q2", personality_name="balanced", research_type="default_research"),
        "Cautious Q2": PersonalityManager(bot_version="q2", personality_name="cautious", research_type="default_research"),
        "Creative Q4": PersonalityManager(bot_version="q4", personality_name="creative", research_type="perplexity_research"),
    }
    
    # Example question data
    question_data = {
        "question": "Will AI safety become a major political issue by 2025?",
        "question_text": "Will AI safety become a major political issue by 2025?",
        "background_info": "AI safety concerns the development of AI systems that are safe and aligned with human values.",
        "resolution_criteria": "This question resolves positively if AI safety becomes a major campaign issue in the 2024 US presidential election.",
        "fine_print": "For this question, 'major political issue' means it is regularly discussed in presidential debates and appears in candidate platforms.",
        "research": "Recent developments suggest growing attention to AI safety concerns among policymakers...",
        "current_date": "2023-06-15"
    }
    
    # Show how each manager produces different prompts
    for name, manager in managers.items():
        print(f"\n{name} Research Prompt (excerpt):")
        research_prompt = manager.get_prompt("research_prompt", question=question_data["question"])
        print("\n".join(research_prompt.split("\n")[:10]))
        print("...\n")
        
        print(f"{name} Binary Forecast Prompt (excerpt):")
        forecast_prompt = manager.get_prompt("binary_forecast_prompt", **question_data)
        print("\n".join(forecast_prompt.split("\n")[:10]))
        print("...\n")
        
        # Show thinking configuration
        print(f"{name} Thinking Configuration:")
        thinking_config = manager.get_thinking_config()
        print(thinking_config)
        print("\n")


if __name__ == "__main__":
    demonstrate_personality_traits()
    demonstrate_personality_configs()
    demonstrate_personality_manager() 