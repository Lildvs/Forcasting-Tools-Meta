"""
Integration tests for personality components and their effects on forecasts.
"""

import unittest
import os
import tempfile
import json
from typing import Dict, Any, Optional

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    ThinkingStyle,
    UncertaintyApproach,
    ReasoningDepth
)
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.data_models.binary_report import BinaryReport


class MockForecaster:
    """Mock forecaster for testing personality effects."""
    
    def __init__(self, personality_name: Optional[str] = None):
        self.personality_name = personality_name
        self.optimizer = PromptOptimizer()
        
        if personality_name:
            # Load personality
            manager = PersonalityManager()
            self.personality = manager.load_personality(personality_name)
        else:
            self.personality = None
    
    def forecast(self, question: BinaryQuestion) -> BinaryReport:
        """
        Create a mock forecast with effects from the personality.
        
        Args:
            question: The question to forecast
            
        Returns:
            A mock forecast report affected by personality
        """
        # Generate a mock probability based on personality traits
        probability = 0.5  # Default neutral
        
        if self.personality:
            # Adjust based on thinking style
            if self.personality.thinking_style == ThinkingStyle.ANALYTICAL:
                probability = 0.6  # More confident
            elif self.personality.thinking_style == ThinkingStyle.CREATIVE:
                probability = 0.7  # Even more confident
                
            # Adjust based on uncertainty approach
            if self.personality.uncertainty_approach == UncertaintyApproach.CAUTIOUS:
                # Move closer to 0.5
                probability = 0.5 + (probability - 0.5) * 0.7
            elif self.personality.uncertainty_approach == UncertaintyApproach.BOLD:
                # Move away from 0.5
                probability = 0.5 + (probability - 0.5) * 1.3
                
            # Clamp between 0.1 and 0.9
            probability = max(0.1, min(0.9, probability))
        
        # Create mock reasoning based on personality
        reasoning = self._generate_mock_reasoning(question)
        
        # Create report
        report = BinaryReport(
            question_text=question.question_text,
            metadata={
                "personality_name": self.personality_name if self.personality_name else "default"
            },
            binary_prob=probability,
            reasoning=reasoning,
            research_report="Mock research report"
        )
        
        return report
    
    def _generate_mock_reasoning(self, question: BinaryQuestion) -> str:
        """Generate mock reasoning based on personality traits."""
        base_reasoning = f"Analysis of the question: '{question.question_text}'"
        
        if not self.personality:
            return f"{base_reasoning}\n\nBased on the available information, I estimate a 50% probability."
        
        # Add personality-specific reasoning
        reasoning = f"{base_reasoning}\n\n"
        
        # Thinking style effects
        if self.personality.thinking_style == ThinkingStyle.ANALYTICAL:
            reasoning += "Taking an analytical approach, I've broken down the key factors:\n"
            reasoning += "- Factor 1: Relevant historical data\n"
            reasoning += "- Factor 2: Current trends\n"
            reasoning += "- Factor 3: Statistical probability\n\n"
        elif self.personality.thinking_style == ThinkingStyle.CREATIVE:
            reasoning += "Taking a creative approach, I've considered multiple scenarios:\n"
            reasoning += "- Scenario 1: The most likely outcome\n"
            reasoning += "- Scenario 2: A plausible alternative\n"
            reasoning += "- Scenario 3: An unexpected but possible outcome\n\n"
        elif self.personality.thinking_style == ThinkingStyle.BAYESIAN:
            reasoning += "Using a Bayesian approach, I've updated my prior beliefs based on the evidence:\n"
            reasoning += "- Prior probability: 50%\n"
            reasoning += "- Evidence strength: Medium\n"
            reasoning += "- Updated probability: See final forecast\n\n"
        else:  # Balanced
            reasoning += "Taking a balanced approach, I've considered multiple perspectives:\n"
            reasoning += "- Historical precedents\n"
            reasoning += "- Expert opinions\n"
            reasoning += "- Current context\n\n"
        
        # Uncertainty approach effects
        if self.personality.uncertainty_approach == UncertaintyApproach.CAUTIOUS:
            reasoning += "Given the uncertainty involved, I'm taking a cautious approach in my forecast.\n\n"
        elif self.personality.uncertainty_approach == UncertaintyApproach.BOLD:
            reasoning += "Despite the uncertainty, I'm confident in identifying the most likely outcome.\n\n"
        else:  # Balanced
            reasoning += "I've balanced confidence with appropriate caution given the uncertainties.\n\n"
        
        # Reasoning depth effects
        if self.personality.reasoning_depth == ReasoningDepth.SHALLOW:
            reasoning += "Based on a quick analysis, my forecast is as follows."
        elif self.personality.reasoning_depth == ReasoningDepth.DEEP:
            reasoning += "After deep analysis of multiple factors and their interactions, my forecast is as follows."
        elif self.personality.reasoning_depth == ReasoningDepth.EXHAUSTIVE:
            reasoning += "Following an exhaustive investigation of all relevant factors, their interactions, and potential scenarios, my forecast is as follows."
        else:  # Moderate
            reasoning += "Based on a thorough analysis of the key factors, my forecast is as follows."
        
        return reasoning


class TestPersonalityIntegration(unittest.TestCase):
    """Integration tests for personality effects on forecasts."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test question
        self.test_question = BinaryQuestion(
            question_text="Will the test pass successfully?",
            background_info="This is a test question for personality integration testing.",
            resolution_criteria="The test passes if all assertions are successful.",
            fine_print="This is a mock test.",
            page_url="",
            api_json={}
        )
        
        # Create temp directory for test personalities
        self.test_dir = tempfile.mkdtemp()
        
        # Create test personalities
        self.test_personalities = {
            "analytical": {
                "name": "analytical",
                "description": "An analytical test personality",
                "thinking_style": "analytical",
                "uncertainty_approach": "balanced",
                "reasoning_depth": "moderate",
                "temperature": 0.5
            },
            "creative": {
                "name": "creative",
                "description": "A creative test personality",
                "thinking_style": "creative",
                "uncertainty_approach": "bold",
                "reasoning_depth": "shallow",
                "temperature": 0.8
            },
            "cautious": {
                "name": "cautious",
                "description": "A cautious test personality",
                "thinking_style": "balanced",
                "uncertainty_approach": "cautious",
                "reasoning_depth": "deep",
                "temperature": 0.4
            },
            "bayesian": {
                "name": "bayesian",
                "description": "A Bayesian test personality",
                "thinking_style": "bayesian",
                "uncertainty_approach": "balanced",
                "reasoning_depth": "exhaustive",
                "temperature": 0.6
            }
        }
        
        # Write test personalities to files
        for name, config in self.test_personalities.items():
            with open(os.path.join(self.test_dir, f"{name}.json"), "w") as f:
                json.dump(config, f)
                
        # Set environment variable to include test directory in personality search paths
        os.environ["PERSONALITY_DIRS"] = self.test_dir

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir)
        
        # Remove environment variable
        if "PERSONALITY_DIRS" in os.environ:
            del os.environ["PERSONALITY_DIRS"]

    def test_personality_effect_on_forecast(self):
        """Test that different personalities produce different forecasts."""
        # Create forecasters with different personalities
        forecaster_analytical = MockForecaster("analytical")
        forecaster_creative = MockForecaster("creative")
        forecaster_cautious = MockForecaster("cautious")
        forecaster_bayesian = MockForecaster("bayesian")
        forecaster_default = MockForecaster()  # No personality
        
        # Generate forecasts
        forecast_analytical = forecaster_analytical.forecast(self.test_question)
        forecast_creative = forecaster_creative.forecast(self.test_question)
        forecast_cautious = forecaster_cautious.forecast(self.test_question)
        forecast_bayesian = forecaster_bayesian.forecast(self.test_question)
        forecast_default = forecaster_default.forecast(self.test_question)
        
        # Check that probabilities differ
        self.assertNotEqual(forecast_analytical.binary_prob, forecast_creative.binary_prob)
        self.assertNotEqual(forecast_analytical.binary_prob, forecast_cautious.binary_prob)
        self.assertNotEqual(forecast_creative.binary_prob, forecast_cautious.binary_prob)
        
        # Check that reasoning differs
        self.assertNotEqual(forecast_analytical.reasoning, forecast_creative.reasoning)
        self.assertNotEqual(forecast_analytical.reasoning, forecast_cautious.reasoning)
        self.assertNotEqual(forecast_creative.reasoning, forecast_cautious.reasoning)
        
        # Check that personality name is in metadata
        self.assertEqual(forecast_analytical.metadata["personality_name"], "analytical")
        self.assertEqual(forecast_creative.metadata["personality_name"], "creative")
        self.assertEqual(forecast_cautious.metadata["personality_name"], "cautious")
        self.assertEqual(forecast_bayesian.metadata["personality_name"], "bayesian")
        self.assertEqual(forecast_default.metadata["personality_name"], "default")

    def test_cautious_vs_bold_uncertainty(self):
        """Test that cautious personalities are more moderate than bold ones."""
        # Create forecasters
        forecaster_cautious = MockForecaster("cautious")
        
        # Create a bold version of the cautious personality
        bold_config = self.test_personalities["cautious"].copy()
        bold_config["name"] = "bold_test"
        bold_config["uncertainty_approach"] = "bold"
        
        with open(os.path.join(self.test_dir, "bold_test.json"), "w") as f:
            json.dump(bold_config, f)
            
        forecaster_bold = MockForecaster("bold_test")
        
        # Generate forecasts
        forecast_cautious = forecaster_cautious.forecast(self.test_question)
        forecast_bold = forecaster_bold.forecast(self.test_question)
        
        # The bold forecast should be further from 0.5 than the cautious one
        cautious_distance = abs(forecast_cautious.binary_prob - 0.5)
        bold_distance = abs(forecast_bold.binary_prob - 0.5)
        
        self.assertLess(cautious_distance, bold_distance)

    def test_reasoning_depth_affects_content(self):
        """Test that reasoning depth affects the content of the reasoning."""
        # Create forecasters with different reasoning depths
        shallow_config = {
            "name": "shallow_test",
            "thinking_style": "balanced",
            "uncertainty_approach": "balanced",
            "reasoning_depth": "shallow"
        }
        
        deep_config = {
            "name": "deep_test",
            "thinking_style": "balanced",
            "uncertainty_approach": "balanced",
            "reasoning_depth": "deep"
        }
        
        exhaustive_config = {
            "name": "exhaustive_test",
            "thinking_style": "balanced",
            "uncertainty_approach": "balanced",
            "reasoning_depth": "exhaustive"
        }
        
        # Write configs to files
        for config in [shallow_config, deep_config, exhaustive_config]:
            with open(os.path.join(self.test_dir, f"{config['name']}.json"), "w") as f:
                json.dump(config, f)
        
        # Create forecasters
        forecaster_shallow = MockForecaster("shallow_test")
        forecaster_deep = MockForecaster("deep_test")
        forecaster_exhaustive = MockForecaster("exhaustive_test")
        
        # Generate forecasts
        forecast_shallow = forecaster_shallow.forecast(self.test_question)
        forecast_deep = forecaster_deep.forecast(self.test_question)
        forecast_exhaustive = forecaster_exhaustive.forecast(self.test_question)
        
        # Check reasoning content reflects depth
        self.assertIn("quick analysis", forecast_shallow.reasoning)
        self.assertIn("deep analysis", forecast_deep.reasoning)
        self.assertIn("exhaustive investigation", forecast_exhaustive.reasoning)
        
        # Check reasoning lengths roughly correspond to depth
        self.assertLess(
            len(forecast_shallow.reasoning),
            len(forecast_deep.reasoning)
        )

    def test_prompt_optimizer_with_personality(self):
        """Test that prompt optimizer correctly incorporates personality traits."""
        optimizer = PromptOptimizer()
        
        # Test generating prompts with different personalities
        prompt_analytical, _ = optimizer.optimize_prompt_pipeline(
            personality_name="analytical",
            template_name="test_template",  # This doesn't exist but we're testing the personality integration
            variables={"question": "Will the test pass?"}
        )
        
        prompt_creative, _ = optimizer.optimize_prompt_pipeline(
            personality_name="creative",
            template_name="test_template",
            variables={"question": "Will the test pass?"}
        )
        
        # Since template doesn't exist, check that personality name is in metadata
        self.assertIn("analytical", str(_))
        self.assertIn("creative", str(_))


class TestPersonalityBenchmark(unittest.TestCase):
    """Performance benchmark tests for personality configurations."""

    def setUp(self):
        """Set up benchmark fixtures."""
        # Create temporary directory for benchmark personalities
        self.benchmark_dir = tempfile.mkdtemp()
        
        # Generate a range of benchmark personalities with different traits
        self.num_personalities = 5  # Number of personalities to generate for each category
        
        # Generate personalities with varying thinking styles
        self.thinking_styles = {"analytical", "creative", "balanced", "bayesian"}
        for style in self.thinking_styles:
            for i in range(self.num_personalities):
                name = f"benchmark_{style}_{i}"
                config = {
                    "name": name,
                    "thinking_style": style,
                    "uncertainty_approach": "balanced",
                    "reasoning_depth": "moderate",
                    "temperature": 0.5 + (i * 0.1)  # Vary temperature
                }
                
                with open(os.path.join(self.benchmark_dir, f"{name}.json"), "w") as f:
                    json.dump(config, f)
        
        # Add benchmark directory to search paths
        os.environ["PERSONALITY_DIRS"] = self.benchmark_dir

    def tearDown(self):
        """Tear down benchmark fixtures."""
        # Clean up
        import shutil
        shutil.rmtree(self.benchmark_dir)
        
        # Remove environment variable
        if "PERSONALITY_DIRS" in os.environ:
            del os.environ["PERSONALITY_DIRS"]

    def test_benchmark_load_performance(self):
        """Benchmark the performance of loading personalities."""
        import time
        manager = PersonalityManager()
        
        # Time loading all personalities
        start_time = time.time()
        personalities = manager.list_available_personalities()
        total_personalities = len(personalities)
        end_time = time.time()
        
        list_time = end_time - start_time
        print(f"\nListed {total_personalities} personalities in {list_time:.4f} seconds")
        
        # Time loading each personality
        load_times = []
        for name in personalities[:10]:  # Test with first 10 to keep test time reasonable
            start_time = time.time()
            manager.load_personality(name)
            end_time = time.time()
            load_times.append(end_time - start_time)
        
        avg_load_time = sum(load_times) / len(load_times)
        print(f"Average load time: {avg_load_time:.4f} seconds per personality")
        
        # Basic performance assertion
        self.assertLess(avg_load_time, 0.1, "Personality loading should be fast")

    def test_benchmark_thinking_styles(self):
        """Benchmark the performance impact of different thinking styles."""
        import time
        
        # Create a test question
        question = BinaryQuestion(
            question_text="Will the benchmark test pass?",
            background_info="This is a benchmark test.",
            resolution_criteria="The test passes if performance is acceptable.",
            fine_print="",
            page_url="",
            api_json={}
        )
        
        # Benchmark different thinking styles
        thinking_style_times = {}
        
        for style in self.thinking_styles:
            # Get personalities with this thinking style
            personalities = [f"benchmark_{style}_{i}" for i in range(self.num_personalities)]
            
            times = []
            for name in personalities:
                forecaster = MockForecaster(name)
                
                # Measure forecast time
                start_time = time.time()
                forecaster.forecast(question)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            # Calculate average time for this thinking style
            avg_time = sum(times) / len(times)
            thinking_style_times[style] = avg_time
            
            print(f"{style.capitalize()} thinking style: {avg_time:.4f} seconds per forecast")
        
        # Log results
        print("\nThinking Style Performance Comparison:")
        for style, avg_time in thinking_style_times.items():
            print(f"- {style.capitalize()}: {avg_time:.4f} seconds")
        
        # No strict assertions as performance will vary by system


if __name__ == "__main__":
    unittest.main() 