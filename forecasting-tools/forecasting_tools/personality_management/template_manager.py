"""
Template Manager with Lazy Loading

This module provides template management for personalities with lazy loading
to improve performance by only loading templates when needed.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Manages templates for personalities with lazy loading capabilities.
    
    Loads templates on-demand and caches them to reduce filesystem operations
    and improve performance.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for the template manager."""
        if cls._instance is None:
            cls._instance = super(TemplateManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the template manager."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._template_paths: Dict[str, str] = {}
        self._template_timestamps: Dict[str, float] = {}
        self._loaded_templates: Set[str] = set()
        self._template_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "templates"),
            os.path.join(os.path.dirname(__file__), "..", "templates", "personalities"),
            "templates",
            "user_templates"
        ]
        logger.debug("Template manager initialized")
    
    def add_template_directory(self, directory: str) -> None:
        """
        Add a directory to search for templates.
        
        Args:
            directory: Directory path to add
        """
        if os.path.isdir(directory) and directory not in self._template_dirs:
            self._template_dirs.append(directory)
            logger.debug(f"Added template directory: {directory}")
    
    def discover_templates(self) -> List[str]:
        """
        Discover available templates without loading them.
        
        Returns:
            List of available template names
        """
        templates = []
        
        for template_dir in self._template_dirs:
            if not os.path.isdir(template_dir):
                continue
                
            for filename in os.listdir(template_dir):
                if filename.endswith('.json') or filename.endswith('.template'):
                    template_name = os.path.splitext(filename)[0]
                    template_path = os.path.join(template_dir, filename)
                    
                    # Store path for lazy loading
                    self._template_paths[template_name] = template_path
                    
                    if template_name not in templates:
                        templates.append(template_name)
        
        logger.debug(f"Discovered {len(templates)} templates")
        return templates
    
    def get_template(self, template_name: str, force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a template, loading it if necessary.
        
        Args:
            template_name: Name of the template to get
            force_reload: Whether to force a reload from disk
            
        Returns:
            Template dictionary or None if not found
        """
        # Check if we need to load the template
        if force_reload or template_name not in self._loaded_templates:
            self._load_template(template_name)
            
        return self._templates.get(template_name)
    
    def _load_template(self, template_name: str) -> bool:
        """
        Load a template from disk.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # If we don't have the path, try to discover templates
        if template_name not in self._template_paths:
            self.discover_templates()
            
        # Check if we have the template path
        if template_name not in self._template_paths:
            logger.warning(f"Template not found: {template_name}")
            return False
            
        template_path = self._template_paths[template_name]
        
        try:
            with open(template_path, 'r') as f:
                template_content = json.load(f)
                
            self._templates[template_name] = template_content
            self._loaded_templates.add(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {str(e)}")
            return False
    
    def get_template_field(self, template_name: str, field_name: str) -> Any:
        """
        Get a specific field from a template.
        
        Args:
            template_name: Name of the template
            field_name: Name of the field to get
            
        Returns:
            Field value or None if not found
        """
        template = self.get_template(template_name)
        if not template:
            return None
            
        return template.get(field_name)
    
    def combine_templates(self, base_template: str, extension_templates: List[str]) -> Dict[str, Any]:
        """
        Combine multiple templates into one.
        
        Args:
            base_template: Name of the base template
            extension_templates: List of templates to extend with
            
        Returns:
            Combined template dictionary
        """
        # Get base template
        combined = self.get_template(base_template)
        if not combined:
            logger.warning(f"Base template not found: {base_template}")
            combined = {}
            
        # Apply extensions
        for ext_name in extension_templates:
            ext_template = self.get_template(ext_name)
            if not ext_template:
                logger.warning(f"Extension template not found: {ext_name}")
                continue
                
            # Merge dictionaries with ext_template taking precedence
            for key, value in ext_template.items():
                if key in combined and isinstance(combined[key], dict) and isinstance(value, dict):
                    # Merge nested dictionaries
                    combined[key].update(value)
                else:
                    # Replace or add value
                    combined[key] = value
        
        return combined
    
    def invalidate_template(self, template_name: str) -> None:
        """
        Invalidate a template in the cache.
        
        Args:
            template_name: Name of the template to invalidate
        """
        if template_name in self._templates:
            del self._templates[template_name]
        if template_name in self._loaded_templates:
            self._loaded_templates.remove(template_name)
        logger.debug(f"Invalidated template: {template_name}")
    
    def invalidate_all_templates(self) -> None:
        """Invalidate all templates in the cache."""
        self._templates.clear()
        self._loaded_templates.clear()
        logger.debug("Invalidated all templates")
    
    def get_template_variables(self, template_name: str) -> Dict[str, Any]:
        """
        Extract variables from a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dictionary of template variables
        """
        template = self.get_template(template_name)
        if not template:
            return {}
            
        # Extract variables section if it exists
        return template.get("variables", {})
        
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Render a template with variables.
        
        Args:
            template_name: Name of the template
            variables: Dictionary of variables to use
            
        Returns:
            Rendered template string or None if template not found
        """
        template = self.get_template(template_name)
        if not template or "content" not in template:
            return None
            
        content = template["content"]
        
        # Replace variables
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            content = content.replace(placeholder, str(var_value))
            
        return content
        
    def apply_template(self, template_name: str, personality_config, **kwargs) -> str:
        """
        Apply a template using a personality configuration.
        
        Args:
            template_name: Name of the template to apply
            personality_config: The personality configuration to use
            **kwargs: Additional variables to use in the template
            
        Returns:
            The rendered template string
        """
        # Try to get the template
        template = self.get_template(template_name)
        
        # If template not found, generate a generic one
        if not template or "content" not in template:
            logger.warning(f"Template not found: {template_name}, using generic fallback")
            
            # Create a reasonable fallback based on the template name
            if "binary_forecast_prompt" in template_name:
                content = """You are a professional forecaster making predictions on a binary question.

Question: {question_text}
Background information: {background_info}
Resolution criteria: {resolution_criteria}
Fine print: {fine_print}

Think carefully through the question step by step and then provide a probability that the answer is Yes. Your answer should be a number between 0.01 (1%) and 0.99 (99%). Choose 0.5 (50%) only if you truly have no information.

The current date is {current_date}.

Consider all relevant factors, historical precedents, current trends, expert opinions, and available evidence.

Please structure your response as follows:
1. Examine the question carefully
2. List key factors to consider
3. Analyze evidence for the Yes outcome
4. Analyze evidence for the No outcome
5. Weigh the evidence
6. Provide your final probability judgment

End your response with a clear probability in the format "Final probability: X%" where X is between 1% and 99%."""
            elif "numeric_forecast_prompt" in template_name:
                content = """You are a professional forecaster making predictions on a numeric question.

Question: {question_text}
Background information: {background_info}
Resolution criteria: {resolution_criteria}
Fine print: {fine_print}
{lower_bound_message}
{upper_bound_message}
Unit of measurement: {unit_of_measure}

The current date is {current_date}.

Think carefully through the question step by step. Consider all relevant factors, historical precedents, current trends, expert opinions, and available evidence.

Please structure your response as follows:
1. Examine the question carefully
2. List key factors to consider
3. Analyze the data and evidence
4. Consider different scenarios (low, medium, high)
5. Provide your forecast as a probability distribution

Your forecast should include estimates at the following percentiles:
- 5th percentile (there's only a 5% chance the actual value will be below this)
- 25th percentile
- 50th percentile (median)
- 75th percentile
- 95th percentile (there's only a 5% chance the actual value will be above this)

End with "Final forecast:" followed by your probability distribution with values at each percentile."""
            elif "multiple_choice_prompt" in template_name:
                content = """You are a professional forecaster making predictions on a multiple-choice question.

Question: {question_text}
Background information: {background_info}
Resolution criteria: {resolution_criteria}
Fine print: {fine_print}
Options: {options}

The current date is {current_date}.

Think carefully through the question step by step. Consider all relevant factors, historical precedents, current trends, expert opinions, and available evidence.

Please structure your response as follows:
1. Examine the question carefully
2. List key factors to consider  
3. Analyze each option systematically
4. Weigh the evidence for each option
5. Provide your probability distribution across all options

For your final forecast, assign a probability to each option. The probabilities must sum to 100%.

End with "Final probabilities:" followed by each option and its percentage chance."""
            else:
                # Generic template
                content = """You are a professional forecaster making predictions.

Question: {question_text}
Background information: {background_info}
Resolution criteria: {resolution_criteria}  
Fine print: {fine_print}

The current date is {current_date}.

Think carefully through the question step by step. Consider all relevant factors, historical precedents, current trends, expert opinions, and available evidence.

Please structure your response as follows:
1. Examine the question carefully
2. List key factors to consider
3. Analyze the evidence
4. Provide your forecast

End with a clear final forecast."""
            
            # Create variables dictionary
            variables = {**kwargs}
            
            # Add personality traits if available
            if personality_config:
                if hasattr(personality_config, "reasoning_depth"):
                    variables["reasoning_depth"] = personality_config.reasoning_depth
                if hasattr(personality_config, "uncertainty_approach"):
                    variables["uncertainty_approach"] = personality_config.uncertainty_approach
                if hasattr(personality_config, "thinking_style"):
                    variables["thinking_style"] = personality_config.thinking_style
                if hasattr(personality_config, "expert_persona"):
                    variables["expert_persona"] = personality_config.expert_persona
            
            # Replace variables
            for var_name, var_value in variables.items():
                placeholder = f"{{{var_name}}}"
                content = content.replace(placeholder, str(var_value))
            
            return content
        
        # Normal case - template exists
        variables = {**kwargs}
        
        # Add personality traits if available  
        if personality_config:
            if hasattr(personality_config, "reasoning_depth"):
                variables["reasoning_depth"] = personality_config.reasoning_depth
            if hasattr(personality_config, "uncertainty_approach"):
                variables["uncertainty_approach"] = personality_config.uncertainty_approach
            if hasattr(personality_config, "thinking_style"):
                variables["thinking_style"] = personality_config.thinking_style
            if hasattr(personality_config, "expert_persona"):
                variables["expert_persona"] = personality_config.expert_persona
                
        content = template["content"]
        
        # Replace variables
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            content = content.replace(placeholder, str(var_value))
            
        return content 