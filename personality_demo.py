#!/usr/bin/env python3
"""
Personality Management Demo Script

This script demonstrates how to use the personality management system
to apply different "personalities" to forecasting prompts.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add forecasting-tools to the Python path
sys.path.append(str(Path(__file__).parent / "forecasting-tools"))

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.personality_management import PersonalityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Sample question data
SAMPLE_QUESTION = {
    "question_text": "Will AI safety become a major political issue by the end of 2025?",
    "background_info": "AI capabilities have been advancing rapidly, with models like GPT-4.1 and Claude 3.7 demonstrating unprecedented capabilities. Various political figures have started expressing concerns about AI safety.",
    "resolution_criteria": "This question resolves positively if AI safety becomes a significant issue in the political campaigns of at least 3 major political parties across G7 countries.",
    "fine_print": "For the purpose of this question, 'major political issue' means that the party has official policy positions on AI safety and candidates regularly discuss it in debates and campaign speeches.",
    "research": "Recent polls show growing public concern about AI safety, with 45% of respondents indicating they are somewhat or very concerned about advanced AI risks. Several politicians have mentioned AI regulation in speeches, though it's not yet a central campaign issue. Tech companies have increased their lobbying related to AI regulation by 30% compared to last year.",
    "current_date": datetime.now().strftime("%Y-%m-%d"),
}

async def demo_personality(personality_name: str, llm: GeneralLlm) -> None:
    """Run a demonstration for a specific personality type."""
    
    logger.info(f"\n{'='*80}\nTesting {personality_name.upper()} personality\n{'='*80}")
    
    # Initialize the personality manager with the specified personality
    personality = PersonalityManager(personality_name=personality_name)
    
    # Get the binary forecast prompt with our sample question data
    prompt = personality.get_prompt(
        "binary_forecast_prompt",
        **SAMPLE_QUESTION
    )
    
    # Get the thinking configuration for this personality
    thinking_config = personality.get_thinking_config()
    logger.info(f"Thinking configuration: {thinking_config}")
    
    # Generate a forecast using the prompt and thinking configuration
    logger.info(f"Generating forecast with {personality_name} personality...")
    response = await llm.invoke(prompt, **thinking_config)
    
    # Print the result
    print(f"\n\nFORECAST WITH {personality_name.upper()} PERSONALITY:\n")
    print(response)
    print("\n" + "="*80 + "\n")

async def main():
    """Run the personality management demo."""
    
    # Create an LLM instance (use Claude if available, otherwise GPT-4.1)
    llm = GeneralLlm(model="anthropic/claude-3-7-sonnet-latest" 
                     if os.getenv("ANTHROPIC_API_KEY") 
                     else "gpt-4.1", 
                     temperature=0.3)
    
    # Demo each personality type
    personalities = ["balanced", "cautious", "creative"]
    for personality in personalities:
        await demo_personality(personality, llm)
        
    # Show available personalities
    all_personalities = PersonalityManager().get_all_personalities()
    logger.info(f"All available personalities: {all_personalities}")

if __name__ == "__main__":
    import os
    import dotenv
    
    # Load environment variables from .env file
    dotenv.load_dotenv()
    
    # Run the demo
    asyncio.run(main()) 