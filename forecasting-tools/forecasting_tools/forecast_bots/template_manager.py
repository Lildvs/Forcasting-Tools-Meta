"""
Template Manager

This module provides a template management system for reasoning approaches,
allowing selection of appropriate templates based on question type and other factors.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Type, Union

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.reasoning import ReasoningApproach
from forecasting_tools.templates.reasoning_templates import (
    APPROACH_TEMPLATES,
    BINARY_QUESTION_BAYESIAN_TEMPLATE,
    NUMERIC_QUESTION_FERMI_TEMPLATE,
    MULTIPLE_CHOICE_TEMPLATE,
    EVIDENCE_EVALUATION_TEMPLATE,
    BAYESIAN_UPDATE_TEMPLATE,
    FERMI_COMPONENT_TEMPLATE,
    UNCERTAINTY_TEMPLATE,
    COGNITIVE_BIAS_TEMPLATE,
    FINAL_FORECAST_TEMPLATE,
    REASONING_STEP_TEMPLATE,
)

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manages templates for different reasoning approaches and question types.
    
    This class:
    - Provides appropriate templates based on question type
    - Allows customization of templates for different reasoning approaches
    - Enables registering new templates and overriding existing ones
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(TemplateManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the template manager."""
        # Skip initialization if already initialized (singleton pattern)
        if TemplateManager._initialized:
            return
        
        TemplateManager._initialized = True
        
        # Start with default templates
        self.templates = {
            "approaches": APPROACH_TEMPLATES.copy(),
            "question_types": {
                BinaryQuestion: {
                    "default": ReasoningApproach.BAYESIAN,
                    "templates": {
                        ReasoningApproach.BAYESIAN: BINARY_QUESTION_BAYESIAN_TEMPLATE,
                        # Other approaches use defaults from APPROACH_TEMPLATES
                    }
                },
                NumericQuestion: {
                    "default": ReasoningApproach.FERMI,
                    "templates": {
                        ReasoningApproach.FERMI: NUMERIC_QUESTION_FERMI_TEMPLATE,
                        # Other approaches use defaults from APPROACH_TEMPLATES
                    }
                },
                MultipleChoiceQuestion: {
                    "default": ReasoningApproach.ANALOG_COMPARISON,
                    "templates": {
                        ReasoningApproach.ANALOG_COMPARISON: MULTIPLE_CHOICE_TEMPLATE,
                        # Other approaches use defaults from APPROACH_TEMPLATES
                    }
                },
                DateQuestion: {
                    "default": ReasoningApproach.TREND_EXTRAPOLATION,
                    "templates": {}  # Use defaults from APPROACH_TEMPLATES
                }
            },
            "components": {
                "evidence_evaluation": EVIDENCE_EVALUATION_TEMPLATE,
                "bayesian_update": BAYESIAN_UPDATE_TEMPLATE,
                "fermi_component": FERMI_COMPONENT_TEMPLATE,
                "uncertainty": UNCERTAINTY_TEMPLATE,
                "cognitive_bias": COGNITIVE_BIAS_TEMPLATE,
                "final_forecast": FINAL_FORECAST_TEMPLATE,
                "reasoning_step": REASONING_STEP_TEMPLATE,
            },
            "custom_templates": {}
        }
        
        # Load any custom templates from files
        self._load_custom_templates()
        
        logger.info("Template manager initialized")
    
    def _load_custom_templates(self) -> None:
        """Load custom templates from template directories."""
        # Paths to check for custom templates
        template_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "templates"),
            os.path.join(os.path.dirname(__file__), "..", "templates", "custom"),
            "templates",
        ]
        
        # Try to find custom templates
        for template_dir in template_dirs:
            template_dir_path = Path(template_dir)
            if not template_dir_path.exists():
                continue
            
            # Look for template files
            for template_file in template_dir_path.glob("*.template"):
                try:
                    with open(template_file, "r") as f:
                        template_content = f.read()
                    
                    # Use filename as template name
                    template_name = template_file.stem
                    self.templates["custom_templates"][template_name] = template_content
                    logger.debug(f"Loaded custom template: {template_name}")
                except Exception as e:
                    logger.warning(f"Error loading template {template_file}: {e}")
    
    def get_default_approach_for_question(
        self, 
        question: Union[MetaculusQuestion, Type[MetaculusQuestion]]
    ) -> ReasoningApproach:
        """
        Get the default reasoning approach for a question type.
        
        Args:
            question: Question instance or question class
            
        Returns:
            Default reasoning approach for this question type
        """
        # Get the question class
        question_class = question.__class__ if isinstance(question, MetaculusQuestion) else question
        
        # Get default approach from templates
        question_config = self.templates["question_types"].get(question_class, {})
        default_approach = question_config.get("default")
        
        if default_approach:
            return default_approach
        
        # Fallback defaults if not configured
        if question_class == BinaryQuestion:
            return ReasoningApproach.BAYESIAN
        elif question_class == NumericQuestion:
            return ReasoningApproach.FERMI
        elif question_class == MultipleChoiceQuestion:
            return ReasoningApproach.ANALOG_COMPARISON
        elif question_class == DateQuestion:
            return ReasoningApproach.TREND_EXTRAPOLATION
        else:
            return ReasoningApproach.SCOUT_MINDSET
    
    def get_template_for_approach(
        self, 
        approach: ReasoningApproach,
        question_type: Optional[Type[MetaculusQuestion]] = None
    ) -> str:
        """
        Get the appropriate template for a reasoning approach.
        
        Args:
            approach: The reasoning approach
            question_type: Optional question type for specialized templates
            
        Returns:
            Template string
        """
        # Check for question-type specific template first
        if question_type and question_type in self.templates["question_types"]:
            question_templates = self.templates["question_types"][question_type]["templates"]
            if approach in question_templates:
                return question_templates[approach]
        
        # Fall back to generic approach template
        approach_key = approach.value
        if approach_key in self.templates["approaches"]:
            return self.templates["approaches"][approach_key]
        
        # If no template found, return an empty string
        logger.warning(f"No template found for approach {approach.value}")
        return ""
    
    def get_component_template(self, component_name: str) -> str:
        """
        Get a component template by name.
        
        Args:
            component_name: Name of the component template
            
        Returns:
            Template string
        """
        return self.templates["components"].get(component_name, "")
    
    def get_custom_template(self, template_name: str) -> str:
        """
        Get a custom template by name.
        
        Args:
            template_name: Name of the custom template
            
        Returns:
            Template string
        """
        return self.templates["custom_templates"].get(template_name, "")
    
    def register_approach_template(
        self, 
        approach: ReasoningApproach, 
        template: str,
        question_type: Optional[Type[MetaculusQuestion]] = None
    ) -> None:
        """
        Register a new template for a reasoning approach.
        
        Args:
            approach: The reasoning approach
            template: The template string
            question_type: Optional question type for specialized templates
        """
        approach_key = approach.value
        
        if question_type:
            # Register for specific question type
            if question_type not in self.templates["question_types"]:
                self.templates["question_types"][question_type] = {
                    "default": approach,
                    "templates": {}
                }
            
            self.templates["question_types"][question_type]["templates"][approach] = template
        else:
            # Register generic approach template
            self.templates["approaches"][approach_key] = template
        
        logger.debug(f"Registered template for approach {approach_key}")
    
    def register_component_template(self, component_name: str, template: str) -> None:
        """
        Register a new component template.
        
        Args:
            component_name: Name of the component
            template: The template string
        """
        self.templates["components"][component_name] = template
        logger.debug(f"Registered component template: {component_name}")
    
    def register_custom_template(self, template_name: str, template: str) -> None:
        """
        Register a new custom template.
        
        Args:
            template_name: Name of the custom template
            template: The template string
        """
        self.templates["custom_templates"][template_name] = template
        logger.debug(f"Registered custom template: {template_name}")
    
    def set_default_approach_for_question(
        self, 
        question_type: Type[MetaculusQuestion],
        approach: ReasoningApproach
    ) -> None:
        """
        Set the default reasoning approach for a question type.
        
        Args:
            question_type: The question class
            approach: The default reasoning approach
        """
        if question_type not in self.templates["question_types"]:
            self.templates["question_types"][question_type] = {
                "default": approach,
                "templates": {}
            }
        else:
            self.templates["question_types"][question_type]["default"] = approach
        
        logger.debug(f"Set default approach for {question_type.__name__} to {approach.value}")
    
    def get_available_approaches(self) -> List[ReasoningApproach]:
        """
        Get a list of all available reasoning approaches.
        
        Returns:
            List of reasoning approaches
        """
        return list(ReasoningApproach)
    
    def get_all_templates(self) -> Dict[str, Any]:
        """
        Get all registered templates.
        
        Returns:
            Dictionary with all templates
        """
        return self.templates
    
    def render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render a template by replacing variables.
        
        Args:
            template: The template string
            variables: Dictionary of variables to replace
            
        Returns:
            Rendered template string
        """
        result = template
        
        # Replace variables
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            if var_value is None:
                var_value = ""
            result = result.replace(placeholder, str(var_value))
        
        return result
    
    def generate_reasoning_template(
        self, 
        question: MetaculusQuestion,
        approach: Optional[ReasoningApproach] = None,
    ) -> Tuple[str, ReasoningApproach]:
        """
        Generate a reasoning template for a specific question.
        
        Args:
            question: The question to generate a template for
            approach: Optional reasoning approach (uses default if not specified)
            
        Returns:
            Tuple of (template string, selected approach)
        """
        # Determine question type
        question_type = question.__class__
        
        # Use provided approach or get default
        selected_approach = approach or self.get_default_approach_for_question(question_type)
        
        # Get appropriate template
        template = self.get_template_for_approach(selected_approach, question_type)
        
        # Populate basic variables
        variables = {
            "question_text": question.question_text,
            "background_info": question.background_info or "No background information provided.",
            "resolution_criteria": question.resolution_criteria or "No specific resolution criteria provided.",
        }
        
        # Add question type specific variables
        if isinstance(question, NumericQuestion):
            variables["units"] = question.unit_of_measure or ""
            if question.has_lower_bound:
                variables["lower_bound"] = str(question.lower_bound)
            if question.has_upper_bound:
                variables["upper_bound"] = str(question.upper_bound)
        
        elif isinstance(question, MultipleChoiceQuestion):
            options = [opt.option for opt in question.options]
            variables["options"] = ", ".join(options)
            variables["option_list"] = "\n".join([f"- {opt}" for opt in options])
        
        # Render the template with basic variables
        rendered_template = self.render_template(template, variables)
        
        return rendered_template, selected_approach 