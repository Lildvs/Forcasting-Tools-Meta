from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBot,
    EnhancedTemplateBotQ1,
    EnhancedTemplateBotQ2,
    EnhancedTemplateBotQ3,
    EnhancedTemplateBotQ4,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot, Notepad
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import Q1TemplateBot2025
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import Q2TemplateBot2025
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import Q3TemplateBot2024
from forecasting_tools.forecast_bots.official_bots.q4_template_bot import Q4TemplateBot2024
from forecasting_tools.forecast_bots.other.uniform_probability_bot import UniformProbabilityBot
from forecasting_tools.forecast_bots.personality_aware_bot import PersonalityAwareBot
from forecasting_tools.forecast_bots.template_bot import TemplateBot
from forecasting_tools.personality_management import PersonalityManager

from typing import Dict, Optional, Type, Union, Any


__all__ = [
    "ForecastBot",
    "Notepad",
    "MainBot",
    "TemplateBot",
    "Q1TemplateBot2025",
    "Q2TemplateBot2025",
    "Q3TemplateBot2024",
    "Q4TemplateBot2024",
    "UniformProbabilityBot",
    "EnhancedTemplateBot",
    "EnhancedTemplateBotQ1",
    "EnhancedTemplateBotQ2",
    "EnhancedTemplateBotQ3",
    "EnhancedTemplateBotQ4",
    "PersonalityAwareBot",
    "create_bot",
    "create_bot_with_personality",
]


BOT_REGISTRY: Dict[str, Type[ForecastBot]] = {
    "main": MainBot,
    "template": TemplateBot,
    "q1_template_2025": Q1TemplateBot2025,
    "q2_template_2025": Q2TemplateBot2025,
    "q3_template_2024": Q3TemplateBot2024,
    "q4_template_2024": Q4TemplateBot2024,
    "uniform": UniformProbabilityBot,
    "enhanced": EnhancedTemplateBot,
    "enhanced_q1": EnhancedTemplateBotQ1,
    "enhanced_q2": EnhancedTemplateBotQ2,
    "enhanced_q3": EnhancedTemplateBotQ3,
    "enhanced_q4": EnhancedTemplateBotQ4,
    "personality_aware": PersonalityAwareBot,
}


def create_bot(
    bot_type: str,
    **kwargs
) -> ForecastBot:
    """
    Create a forecast bot of the specified type.
    
    Args:
        bot_type: The type of bot to create (see BOT_REGISTRY for options)
        **kwargs: Additional arguments to pass to the bot constructor
        
    Returns:
        A forecast bot instance
        
    Raises:
        ValueError: If the bot type is not recognized
    """
    if bot_type not in BOT_REGISTRY:
        raise ValueError(
            f"Unknown bot type: {bot_type}. Available types: {list(BOT_REGISTRY.keys())}"
        )
    
    bot_class = BOT_REGISTRY[bot_type]
    return bot_class(**kwargs)


def create_bot_with_personality(
    bot_type: str = "personality_aware",
    personality_name: str = "balanced",
    research_type: str = "perplexity",
    **kwargs
) -> ForecastBot:
    """
    Create a forecast bot with a specific personality.
    
    Args:
        bot_type: The type of bot to create (defaults to "personality_aware")
        personality_name: The name of the personality to use
        research_type: The type of research to use
        **kwargs: Additional arguments to pass to the bot constructor
        
    Returns:
        A forecast bot instance configured with the specified personality
        
    Raises:
        ValueError: If the bot type is not recognized or the personality is not found
    """
    # Verify that the personality exists
    try:
        # Create a temporary personality manager to check if the personality exists
        PersonalityManager(personality_name=personality_name)
    except FileNotFoundError:
        available_personalities = PersonalityManager().get_all_personalities()
        raise ValueError(
            f"Unknown personality: {personality_name}. "
            f"Available personalities: {available_personalities}"
        )
    
    # Create the bot
    if bot_type == "personality_aware":
        return PersonalityAwareBot(
            personality_name=personality_name,
            research_type=research_type,
            **kwargs
        )
    else:
        # For other bot types, just add the personality name to kwargs
        return create_bot(
            bot_type,
            personality_name=personality_name,
            **kwargs
        )
