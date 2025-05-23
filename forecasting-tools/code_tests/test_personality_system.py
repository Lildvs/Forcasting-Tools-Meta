"""
Test Personality System

This module provides tests for the personality management system and
demonstrates how different personalities affect forecasts.
"""

import asyncio
import os
import unittest
from datetime import datetime
from typing import Dict, List, Optional

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots import create_bot_with_personality
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig


# Sample question for testing
SAMPLE_BINARY_QUESTION = BinaryQuestion(
    id=0,
    title="AI Safety Political Issue",
    question_text="Will AI safety become a major political issue by the end of 2025?",
    background_info="AI capabilities have been advancing rapidly, with models like GPT-4 and Claude 3 demonstrating unprecedented capabilities. Various political figures have started expressing concerns about AI safety.",
    resolution_criteria="This question resolves positively if AI safety becomes a significant issue in the political campaigns of at least 3 major political parties across G7 countries.",
    fine_print="For the purpose of this question, 'major political issue' means that the party has official policy positions on AI safety and candidates regularly discuss it in debates and campaign speeches.",
    publish_time=datetime.now(),
    resolve_time=datetime(2025, 12, 31),
    resolution=None,
    url="https://example.com/questions/ai-safety-politics",
    possibilities=None,
)


class TestPersonalitySystem(unittest.TestCase):
    """Test the personality management system."""
    
    def setUp(self):
        """Set up the test case."""
        # Skip tests if API keys aren't available
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            self.skipTest("No OpenAI or Anthropic API key available")
    
    def test_personality_config_loading(self):
        """Test that personality configurations can be loaded."""
        # Initialize personality manager
        manager = PersonalityManager()
        
        # Get available personalities
        personalities = manager.get_all_personalities()
        
        # Ensure that our personalities are available
        required_personalities = ["balanced", "cautious", "creative", "economist", "bayesian"]
        for personality in required_personalities:
            self.assertIn(personality, personalities)
        
        # Test loading each personality
        for personality in required_personalities:
            config = manager.load_personality(personality)
            self.assertIsInstance(config, PersonalityConfig)
            self.assertEqual(config.name, personality)
    
    def test_personality_prompt_generation(self):
        """Test that prompts can be generated with different personalities."""
        # Test with each personality
        for personality_name in ["balanced", "cautious", "creative", "economist", "bayesian"]:
            manager = PersonalityManager(personality_name=personality_name)
            
            # Generate binary forecast prompt
            prompt = manager.get_prompt(
                "binary_forecast_prompt",
                question_text=SAMPLE_BINARY_QUESTION.question_text,
                background_info=SAMPLE_BINARY_QUESTION.background_info,
                resolution_criteria=SAMPLE_BINARY_QUESTION.resolution_criteria,
                fine_print=SAMPLE_BINARY_QUESTION.fine_print,
                research="Recent polls show growing public concern about AI safety.",
                current_date=datetime.now().strftime("%Y-%m-%d"),
            )
            
            # Check that the prompt contains personality-specific content
            self.assertIn(personality_name, manager.get_current_personality().name)
            self.assertIn("QUESTION:", prompt)
            self.assertIn(SAMPLE_BINARY_QUESTION.question_text, prompt)
    
    def test_thinking_config_differences(self):
        """Test that different personalities produce different thinking configurations."""
        thinking_configs = {}
        
        # Get thinking configs for each personality
        for personality_name in ["balanced", "cautious", "creative", "economist", "bayesian"]:
            manager = PersonalityManager(personality_name=personality_name)
            thinking_configs[personality_name] = manager.get_thinking_config()
        
        # Check that there are some differences between thinking configs
        self.assertNotEqual(
            thinking_configs["cautious"],
            thinking_configs["creative"]
        )
        
        # Cautious should have lower temperature than creative
        self.assertLess(
            thinking_configs["cautious"]["temperature"],
            thinking_configs["creative"]["temperature"]
        )


class TestPersonalityAwareBot(unittest.TestCase):
    """Test the PersonalityAwareBot."""
    
    def setUp(self):
        """Set up the test case."""
        # Skip tests if API keys aren't available
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            self.skipTest("No OpenAI or Anthropic API key available")
        
        # Use a smaller model for testing
        self.test_llm = GeneralLlm(model="gpt-3.5-turbo", temperature=0.2)
    
    async def _forecast_with_personality(self, personality_name: str) -> float:
        """
        Make a forecast with a specific personality.
        
        Args:
            personality_name: The name of the personality to use
            
        Returns:
            The prediction value (0-1)
        """
        # Create a bot with the specified personality
        bot = create_bot_with_personality(
            personality_name=personality_name,
            llms={"default": self.test_llm, "research": self.test_llm}
        )
        
        # Run a simple research string for testing
        research = f"Research conducted using the {personality_name} personality approach."
        
        # Make a prediction
        prediction = await bot._run_forecast_on_binary(SAMPLE_BINARY_QUESTION, research)
        
        return prediction.prediction_value
    
    def test_different_personalities_produce_different_forecasts(self):
        """Test that different personalities produce different forecasts."""
        # Run the test asynchronously
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._run_forecast_comparison())
        
        # Check that at least some forecasts are different
        predictions = list(results.values())
        self.assertTrue(any(p1 != p2 for p1 in predictions for p2 in predictions if p1 != p2))
        
        # Print the results
        print("\nForecast comparison:")
        for personality, prediction in results.items():
            print(f"{personality}: {prediction:.2%}")
    
    async def _run_forecast_comparison(self) -> Dict[str, float]:
        """
        Run forecasts with different personalities.
        
        Returns:
            Dictionary mapping personality names to prediction values
        """
        personalities = ["balanced", "cautious", "creative", "economist", "bayesian"]
        results = {}
        
        for personality in personalities:
            try:
                prediction = await self._forecast_with_personality(personality)
                results[personality] = prediction
            except Exception as e:
                print(f"Error forecasting with {personality}: {e}")
        
        return results


if __name__ == "__main__":
    unittest.main() 