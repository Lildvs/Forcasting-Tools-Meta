"""
Automated tests for the personality management system in CI/CD pipelines.

These tests verify the core functionality of the personality system
and are designed to run as part of continuous integration workflows.
"""

import os
import unittest
import tempfile
import shutil
import json
from typing import Dict, Any, List, Optional

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig, ThinkingStyle, UncertaintyApproach, ReasoningDepth
from forecasting_tools.personality_management.template_manager import TemplateManager
from forecasting_tools.personality_management.cache import PersonalityCache
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer
from forecasting_tools.personality_management.validators import PersonalityValidator
from forecasting_tools.personality_management.health_check import check_system_health


class TestPersonalitySystem(unittest.TestCase):
    """Test suite for personality management system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create a temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test personality and template directories
        cls.personality_dir = os.path.join(cls.test_dir, "personalities")
        cls.template_dir = os.path.join(cls.test_dir, "templates")
        os.makedirs(cls.personality_dir, exist_ok=True)
        os.makedirs(cls.template_dir, exist_ok=True)
        
        # Create test personality file
        cls.create_test_personality()
        
        # Create test template file
        cls.create_test_template()
        
        # Initialize components
        cls.manager = PersonalityManager()
        cls.manager.add_personality_directory(cls.personality_dir)
        
        cls.template_manager = TemplateManager()
        cls.template_manager.add_template_directory(cls.template_dir)
        
        cls.cache = PersonalityCache()
        cls.optimizer = PromptOptimizer()
        cls.validator = PersonalityValidator()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_test_personality(cls):
        """Create a test personality configuration file."""
        test_personality = {
            "name": "test_analytical",
            "description": "Test analytical personality",
            "thinking_style": "analytical",
            "uncertainty_approach": "balanced",
            "reasoning_depth": "moderate",
            "temperature": 0.6,
            "traits": {
                "precision": {
                    "name": "precision",
                    "description": "Precision in analysis",
                    "value": 0.8
                }
            },
            "template_variables": {
                "reasoning_approach": "systematic"
            }
        }
        
        # Write to file
        file_path = os.path.join(cls.personality_dir, "test_analytical.json")
        with open(file_path, 'w') as f:
            json.dump(test_personality, f, indent=2)
        
        return file_path
    
    @classmethod
    def create_test_template(cls):
        """Create a test template file."""
        test_template = {
            "content": "Consider the following question: {{question}}\n\nApproach this with a {{thinking_style}} thinking style and a {{uncertainty_approach}} approach to uncertainty. Reasoning should be {{reasoning_depth}}.\n\n{{#reasoning_approach}}Use a {{reasoning_approach}} reasoning approach.{{/reasoning_approach}}",
            "variables": {
                "question": "",
                "thinking_style": "balanced",
                "uncertainty_approach": "balanced",
                "reasoning_depth": "moderate",
                "reasoning_approach": None
            }
        }
        
        # Write to file
        file_path = os.path.join(cls.template_dir, "test_template.json")
        with open(file_path, 'w') as f:
            json.dump(test_template, f, indent=2)
        
        return file_path
    
    def test_personality_loading(self):
        """Test that personalities can be loaded."""
        # List available personalities
        personalities = self.manager.list_available_personalities()
        
        # Check if our test personality is available
        self.assertIn("test_analytical", personalities)
        
        # Load the personality
        personality = self.manager.load_personality("test_analytical")
        
        # Verify it loaded correctly
        self.assertIsNotNone(personality)
        self.assertEqual(personality.name, "test_analytical")
        self.assertEqual(personality.thinking_style, ThinkingStyle.ANALYTICAL)
        self.assertEqual(personality.uncertainty_approach, UncertaintyApproach.BALANCED)
        self.assertEqual(personality.reasoning_depth, ReasoningDepth.MODERATE)
        self.assertEqual(personality.temperature, 0.6)
        
        # Check traits
        self.assertIn("precision", personality.traits)
        self.assertEqual(personality.traits["precision"].value, 0.8)
        
        # Check template variables
        self.assertIn("reasoning_approach", personality.template_variables)
        self.assertEqual(personality.template_variables["reasoning_approach"], "systematic")
    
    def test_personality_creation(self):
        """Test creating a personality from a dictionary."""
        # Create a personality config
        config_dict = {
            "name": "test_creative",
            "description": "Test creative personality",
            "thinking_style": "creative",
            "uncertainty_approach": "bold",
            "reasoning_depth": "shallow",
            "temperature": 0.8
        }
        
        # Create personality from dict
        personality = PersonalityConfig.from_dict(config_dict)
        
        # Verify it was created correctly
        self.assertEqual(personality.name, "test_creative")
        self.assertEqual(personality.thinking_style, ThinkingStyle.CREATIVE)
        self.assertEqual(personality.uncertainty_approach, UncertaintyApproach.BOLD)
        self.assertEqual(personality.reasoning_depth, ReasoningDepth.SHALLOW)
        self.assertEqual(personality.temperature, 0.8)
        
        # Register it
        self.manager.register_personality(personality)
        
        # Check if it was registered
        personalities = self.manager.list_available_personalities()
        self.assertIn("test_creative", personalities)
    
    def test_template_loading(self):
        """Test that templates can be loaded."""
        # Discover templates
        templates = self.template_manager.discover_templates()
        
        # Check if our test template is available
        self.assertIn("test_template", templates)
        
        # Load the template
        template = self.template_manager.get_template("test_template")
        
        # Verify it loaded correctly
        self.assertIsNotNone(template)
        self.assertIn("content", template)
        self.assertIn("variables", template)
    
    def test_template_rendering(self):
        """Test template rendering."""
        # Variables for rendering
        variables = {
            "question": "Will the stock market rise tomorrow?",
            "thinking_style": "analytical",
            "uncertainty_approach": "cautious",
            "reasoning_depth": "deep",
            "reasoning_approach": "systematic"
        }
        
        # Render the template
        rendered = self.template_manager.render_template("test_template", variables)
        
        # Verify output contains expected text
        self.assertIsNotNone(rendered)
        self.assertIn("Will the stock market rise tomorrow?", rendered)
        self.assertIn("analytical thinking style", rendered)
        self.assertIn("cautious approach to uncertainty", rendered)
        self.assertIn("Reasoning should be deep", rendered)
        self.assertIn("Use a systematic reasoning approach", rendered)
    
    def test_cache_functionality(self):
        """Test cache functionality."""
        # Clear cache first
        self.cache.invalidate_all()
        
        # Get initial stats
        pre_stats = self.cache.get_stats()
        initial_hits = pre_stats.get("hits", 0)
        
        # Load a personality (should add to cache)
        self.manager.load_personality("test_analytical")
        
        # Load again (should use cache)
        self.manager.load_personality("test_analytical")
        
        # Get stats after test
        post_stats = self.cache.get_stats()
        final_hits = post_stats.get("hits", 0)
        
        # Should have at least one more hit
        self.assertGreater(final_hits, initial_hits)
    
    def test_prompt_generation(self):
        """Test prompt generation."""
        # Generate a prompt
        prompt, metadata = self.optimizer.optimize_prompt_pipeline(
            personality_name="test_analytical",
            template_name="test_template",
            variables={"question": "Will AI significantly impact future job markets?"}
        )
        
        # Verify prompt was generated
        self.assertIsNotNone(prompt)
        self.assertGreater(len(prompt), 20)
        
        # Verify metadata
        self.assertIsNotNone(metadata)
        self.assertIn("estimated_tokens", metadata)
    
    def test_prompt_caching(self):
        """Test prompt caching."""
        # Clear prompt cache
        self.optimizer.clear_cache()
        
        # Generate a prompt for the first time
        prompt1, _ = self.optimizer.optimize_prompt_pipeline(
            personality_name="test_analytical",
            template_name="test_template",
            variables={"question": "Will AI reach human-level intelligence within 10 years?"}
        )
        
        # Get cache stats after first generation
        cache_stats1 = self.optimizer.get_cache_stats()
        
        # Generate the same prompt again
        prompt2, _ = self.optimizer.optimize_prompt_pipeline(
            personality_name="test_analytical",
            template_name="test_template",
            variables={"question": "Will AI reach human-level intelligence within 10 years?"}
        )
        
        # Get cache stats after second generation
        cache_stats2 = self.optimizer.get_cache_stats()
        
        # Verify prompts are the same
        self.assertEqual(prompt1, prompt2)
        
        # Verify cache hits increased
        self.assertGreater(cache_stats2.get("hits", 0), cache_stats1.get("hits", 0))
    
    def test_personality_validation(self):
        """Test personality validation."""
        # Get a valid personality
        valid_personality = self.manager.load_personality("test_analytical")
        
        # Validate it
        is_valid, errors = self.validator.validate_personality(valid_personality)
        
        # Should be valid
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Create an invalid personality
        invalid_config = {
            "name": "invalid_personality",
            # Missing required fields
            "thinking_style": "invalid_style",  # Invalid value
            "uncertainty_approach": "balanced",
            "reasoning_depth": "moderate"
        }
        
        invalid_personality = PersonalityConfig.from_dict(invalid_config)
        
        # Validate it
        is_valid, errors = self.validator.validate_personality(invalid_personality)
        
        # Should be invalid
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_template_compatibility(self):
        """Test template compatibility checking."""
        # Get a personality
        personality = self.manager.load_personality("test_analytical")
        
        # Check compatibility with template
        is_compatible, issues = self.validator.validate_template_compatibility(
            personality, "test_template"
        )
        
        # Should be compatible
        self.assertTrue(is_compatible)
        self.assertEqual(len(issues), 0)
    
    def test_system_health_check(self):
        """Test the system health check."""
        # Run a minimal health check
        results = check_system_health("minimal")
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn("overall_status", results)
        self.assertIn("components", results)
        
        # Should have at least core components
        self.assertIn("personality_loading", results["components"])
        self.assertIn("template_loading", results["components"])


class TestPersonalitySystemEdgeCases(unittest.TestCase):
    """Test suite for edge cases in the personality management system."""
    
    def setUp(self):
        """Set up each test."""
        self.manager = PersonalityManager()
        self.template_manager = TemplateManager()
        self.validator = PersonalityValidator()
    
    def test_nonexistent_personality(self):
        """Test loading a nonexistent personality."""
        # Try to load a nonexistent personality
        personality = self.manager.load_personality("nonexistent_personality")
        
        # Should return None
        self.assertIsNone(personality)
    
    def test_nonexistent_template(self):
        """Test loading a nonexistent template."""
        # Try to load a nonexistent template
        template = self.template_manager.get_template("nonexistent_template")
        
        # Should return None
        self.assertIsNone(template)
    
    def test_template_without_required_variable(self):
        """Test rendering a template with missing required variable."""
        # Find an existing template
        templates = self.template_manager.discover_templates()
        if not templates:
            self.skipTest("No templates available for testing")
        
        test_template = templates[0]
        
        # Try rendering with empty variables
        rendered = self.template_manager.render_template(test_template, {})
        
        # Should still return something (albeit with potentially empty variables)
        self.assertIsNotNone(rendered)
    
    def test_personality_with_invalid_enum_values(self):
        """Test creating a personality with invalid enum values."""
        # Create a personality config with invalid enum values
        config_dict = {
            "name": "invalid_enums",
            "description": "Invalid enum values",
            "thinking_style": "invalid_style",
            "uncertainty_approach": "invalid_approach",
            "reasoning_depth": "invalid_depth",
            "temperature": 0.7
        }
        
        # Create personality from dict - should not raise exception but use default values
        personality = PersonalityConfig.from_dict(config_dict)
        
        # Validate it - should be invalid
        is_valid, errors = self.validator.validate_personality(personality)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


if __name__ == '__main__':
    unittest.main() 