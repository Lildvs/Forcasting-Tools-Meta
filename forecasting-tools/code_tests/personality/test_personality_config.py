"""
Unit tests for the personality configuration module.
"""

import unittest
import os
import tempfile
import json
from typing import Dict, Any

from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    ThinkingStyle,
    UncertaintyApproach,
    ReasoningDepth,
    PersonalityTrait
)


class TestPersonalityConfig(unittest.TestCase):
    """Test cases for PersonalityConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            "name": "test_personality",
            "description": "A test personality",
            "thinking_style": "analytical",
            "uncertainty_approach": "cautious",
            "reasoning_depth": "deep",
            "temperature": 0.5,
            "traits": {
                "creativity": {
                    "name": "creativity",
                    "description": "Level of creative thinking",
                    "value": 0.3
                },
                "thoroughness": {
                    "name": "thoroughness",
                    "description": "Attention to detail",
                    "value": 0.8
                }
            },
            "template_variables": {
                "custom_var": "custom value",
                "extra_instructions": "Be precise"
            }
        }

    def test_create_from_dict(self):
        """Test creating a personality from a dictionary."""
        config = PersonalityConfig.from_dict(self.sample_config)
        
        # Check basic properties
        self.assertEqual(config.name, "test_personality")
        self.assertEqual(config.description, "A test personality")
        self.assertEqual(config.thinking_style, ThinkingStyle.ANALYTICAL)
        self.assertEqual(config.uncertainty_approach, UncertaintyApproach.CAUTIOUS)
        self.assertEqual(config.reasoning_depth, ReasoningDepth.DEEP)
        self.assertEqual(config.temperature, 0.5)
        
        # Check traits
        self.assertIn("creativity", config.traits)
        self.assertIn("thoroughness", config.traits)
        self.assertEqual(config.traits["creativity"].value, 0.3)
        self.assertEqual(config.traits["thoroughness"].value, 0.8)
        
        # Check template variables
        self.assertEqual(config.template_variables["custom_var"], "custom value")
        self.assertEqual(config.template_variables["extra_instructions"], "Be precise")

    def test_invalid_thinking_style(self):
        """Test handling of invalid thinking style."""
        config_data = self.sample_config.copy()
        config_data["thinking_style"] = "invalid_style"
        
        with self.assertRaises(ValueError):
            PersonalityConfig.from_dict(config_data)

    def test_invalid_uncertainty_approach(self):
        """Test handling of invalid uncertainty approach."""
        config_data = self.sample_config.copy()
        config_data["uncertainty_approach"] = "invalid_approach"
        
        with self.assertRaises(ValueError):
            PersonalityConfig.from_dict(config_data)

    def test_invalid_reasoning_depth(self):
        """Test handling of invalid reasoning depth."""
        config_data = self.sample_config.copy()
        config_data["reasoning_depth"] = "invalid_depth"
        
        with self.assertRaises(ValueError):
            PersonalityConfig.from_dict(config_data)

    def test_missing_required_field(self):
        """Test handling of missing required field."""
        config_data = self.sample_config.copy()
        del config_data["name"]
        
        with self.assertRaises(ValueError):
            PersonalityConfig.from_dict(config_data)

    def test_to_dict(self):
        """Test converting a personality to a dictionary."""
        config = PersonalityConfig.from_dict(self.sample_config)
        config_dict = config.to_dict()
        
        # Check fields are preserved
        self.assertEqual(config_dict["name"], "test_personality")
        self.assertEqual(config_dict["thinking_style"], "analytical")
        self.assertEqual(config_dict["uncertainty_approach"], "cautious")
        self.assertEqual(config_dict["reasoning_depth"], "deep")
        self.assertEqual(config_dict["temperature"], 0.5)
        
        # Check traits
        self.assertIn("creativity", config_dict["traits"])
        self.assertEqual(config_dict["traits"]["creativity"]["value"], 0.3)

    def test_save_load_json(self):
        """Test saving and loading a personality to/from JSON."""
        config = PersonalityConfig.from_dict(self.sample_config)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
            temp_path = temp.name
            
            # Save to file
            config_dict = config.to_dict()
            json.dump(config_dict, temp)
        
        try:
            # Load from file
            with open(temp_path, "r") as f:
                loaded_dict = json.load(f)
            
            loaded_config = PersonalityConfig.from_dict(loaded_dict)
            
            # Check equality
            self.assertEqual(loaded_config.name, config.name)
            self.assertEqual(loaded_config.thinking_style, config.thinking_style)
            self.assertEqual(loaded_config.uncertainty_approach, config.uncertainty_approach)
            self.assertEqual(len(loaded_config.traits), len(config.traits))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_optional_fields(self):
        """Test handling of optional fields."""
        # Remove optional fields
        config_data = {
            "name": "minimal_config",
            "thinking_style": "balanced",
            "uncertainty_approach": "balanced",
            "reasoning_depth": "moderate"
        }
        
        config = PersonalityConfig.from_dict(config_data)
        
        # Check defaults
        self.assertIsNone(config.description)
        self.assertEqual(config.temperature, 0.7)  # Default
        self.assertEqual(len(config.traits), 0)
        self.assertEqual(len(config.template_variables), 0)

    def test_trait_access(self):
        """Test accessing traits directly."""
        config = PersonalityConfig.from_dict(self.sample_config)
        
        # Access existing trait
        trait = config.get_trait("creativity")
        self.assertIsNotNone(trait)
        self.assertEqual(trait.value, 0.3)
        
        # Access non-existent trait
        trait = config.get_trait("nonexistent")
        self.assertIsNone(trait)

    def test_trait_validation(self):
        """Test trait value validation."""
        # Test with invalid trait type
        config_data = self.sample_config.copy()
        config_data["traits"]["invalid_trait"] = {
            "name": "invalid_trait",
            "description": "Invalid trait",
            "value": "not_a_valid_type"  # Should be numeric or string
        }
        
        # This should still work as traits are flexible
        config = PersonalityConfig.from_dict(config_data)
        self.assertEqual(config.traits["invalid_trait"].value, "not_a_valid_type")


class TestPersonalityTrait(unittest.TestCase):
    """Test cases for PersonalityTrait class."""

    def test_create_trait(self):
        """Test creating a personality trait."""
        trait = PersonalityTrait(
            name="creativity",
            description="Level of creative thinking",
            value=0.8
        )
        
        self.assertEqual(trait.name, "creativity")
        self.assertEqual(trait.description, "Level of creative thinking")
        self.assertEqual(trait.value, 0.8)

    def test_trait_to_dict(self):
        """Test converting a trait to a dictionary."""
        trait = PersonalityTrait(
            name="creativity",
            description="Level of creative thinking",
            value=0.8
        )
        
        trait_dict = trait.to_dict()
        
        self.assertEqual(trait_dict["name"], "creativity")
        self.assertEqual(trait_dict["description"], "Level of creative thinking")
        self.assertEqual(trait_dict["value"], 0.8)

    def test_trait_from_dict(self):
        """Test creating a trait from a dictionary."""
        trait_dict = {
            "name": "creativity",
            "description": "Level of creative thinking",
            "value": 0.8
        }
        
        trait = PersonalityTrait.from_dict(trait_dict)
        
        self.assertEqual(trait.name, "creativity")
        self.assertEqual(trait.description, "Level of creative thinking")
        self.assertEqual(trait.value, 0.8)

    def test_trait_missing_fields(self):
        """Test handling of missing fields in trait dictionary."""
        trait_dict = {
            "name": "creativity",
            "value": 0.8
        }
        
        trait = PersonalityTrait.from_dict(trait_dict)
        
        self.assertEqual(trait.name, "creativity")
        self.assertEqual(trait.value, 0.8)
        self.assertEqual(trait.description, "")  # Default empty string


if __name__ == "__main__":
    unittest.main() 