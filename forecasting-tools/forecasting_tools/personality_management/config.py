"""
Personality Configuration Module

This module defines the data models for personality traits and configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union


class ReasoningDepth(str, Enum):
    """Enum for reasoning depth levels."""
    SHALLOW = "shallow"
    MODERATE = "moderate"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class UncertaintyApproach(str, Enum):
    """Enum for approaches to handling uncertainty."""
    OVERCONFIDENT = "overconfident"
    BALANCED = "balanced"
    CAUTIOUS = "cautious"
    EXPLICIT = "explicit"  # Explicitly quantifies all uncertainties


class ThinkingStyle(str, Enum):
    """Enum for thinking styles."""
    INTUITIVE = "intuitive"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    BAYESIAN = "bayesian"
    FERMI = "fermi"


@dataclass
class PersonalityTrait:
    """Base class for a personality trait."""
    name: str
    description: str
    value: Union[str, int, float]


@dataclass
class PersonalityConfig:
    """
    Data model for personality configuration.
    
    Attributes:
        name: The name of the personality.
        description: A description of the personality.
        reasoning_depth: How deeply the personality reasons through problems.
        uncertainty_approach: How the personality handles uncertainty.
        expert_persona: The domain expertise persona to adopt.
        thinking_style: The style of thinking to employ.
        temperature: The LLM temperature setting (0.0-1.0).
        traits: Additional custom traits for this personality.
        template_variables: Variables to inject into templates.
    """
    name: str
    description: str
    reasoning_depth: ReasoningDepth = ReasoningDepth.MODERATE
    uncertainty_approach: UncertaintyApproach = UncertaintyApproach.BALANCED
    expert_persona: Optional[str] = None
    thinking_style: ThinkingStyle = ThinkingStyle.ANALYTICAL
    temperature: float = 0.7
    traits: Dict[str, PersonalityTrait] = field(default_factory=dict)
    template_variables: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "PersonalityConfig":
        """Create a PersonalityConfig from a dictionary."""
        # Handle enums
        if "reasoning_depth" in data:
            data["reasoning_depth"] = ReasoningDepth(data["reasoning_depth"])
        if "uncertainty_approach" in data:
            data["uncertainty_approach"] = UncertaintyApproach(data["uncertainty_approach"])
        if "thinking_style" in data:
            data["thinking_style"] = ThinkingStyle(data["thinking_style"])
            
        # Handle traits
        if "traits" in data:
            traits = {}
            for name, trait_data in data["traits"].items():
                traits[name] = PersonalityTrait(
                    name=name,
                    description=trait_data.get("description", ""),
                    value=trait_data.get("value")
                )
            data["traits"] = traits
            
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "reasoning_depth": self.reasoning_depth.value,
            "uncertainty_approach": self.uncertainty_approach.value,
            "thinking_style": self.thinking_style.value,
            "temperature": self.temperature,
        }
        
        if self.expert_persona:
            result["expert_persona"] = self.expert_persona
            
        if self.traits:
            result["traits"] = {
                name: {
                    "description": trait.description,
                    "value": trait.value
                } for name, trait in self.traits.items()
            }
            
        if self.template_variables:
            result["template_variables"] = self.template_variables
            
        return result
    
    def get_thinking_config(self) -> Dict:
        """Get LLM thinking configuration based on personality."""
        config = {"temperature": self.temperature}
        
        # Add more parameters based on personality traits
        if self.reasoning_depth == ReasoningDepth.DEEP:
            config["max_tokens"] = 4000
        elif self.reasoning_depth == ReasoningDepth.EXHAUSTIVE:
            config["max_tokens"] = 8000
            
        return config 