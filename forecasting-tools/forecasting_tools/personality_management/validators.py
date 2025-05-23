"""
Validation Utilities for Personality Management

This module provides utilities for validating personality configurations,
checking for consistency, and ensuring proper compatibility.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import re

from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    ThinkingStyle,
    UncertaintyApproach,
    ReasoningDepth,
    PersonalityTrait
)
from forecasting_tools.personality_management.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class PersonalityValidator:
    """
    Validator for personality configurations.
    
    This class provides methods to validate personality configurations,
    check for consistency, and verify compatibility with the system.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.template_manager = TemplateManager()
    
    def validate_personality(
        self, 
        personality: Union[PersonalityConfig, Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a personality configuration.
        
        Args:
            personality: Personality configuration object or dictionary
            
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        errors = []
        
        # Convert dict to PersonalityConfig if needed
        config = None
        if isinstance(personality, dict):
            try:
                config = PersonalityConfig.from_dict(personality)
            except Exception as e:
                errors.append(f"Failed to parse personality configuration: {str(e)}")
                return False, errors
        else:
            config = personality
        
        # Check required fields
        if not config.name:
            errors.append("Personality name is required")
        
        # Check valid enum values
        try:
            ThinkingStyle(config.thinking_style)
        except ValueError:
            errors.append(f"Invalid thinking style: {config.thinking_style}")
            
        try:
            UncertaintyApproach(config.uncertainty_approach)
        except ValueError:
            errors.append(f"Invalid uncertainty approach: {config.uncertainty_approach}")
            
        try:
            ReasoningDepth(config.reasoning_depth)
        except ValueError:
            errors.append(f"Invalid reasoning depth: {config.reasoning_depth}")
        
        # Check temperature range
        if hasattr(config, "temperature"):
            if not isinstance(config.temperature, (int, float)):
                errors.append(f"Temperature must be a number, got {type(config.temperature).__name__}")
            elif config.temperature < 0 or config.temperature > 2:
                errors.append(f"Temperature should be between 0 and 2, got {config.temperature}")
        
        # Check traits
        for name, trait in config.traits.items():
            # Name consistency
            if trait.name != name:
                errors.append(f"Trait name mismatch: key is '{name}' but trait name is '{trait.name}'")
            
            # Check trait values are of appropriate types
            if not isinstance(trait.value, (int, float, str, bool)):
                errors.append(f"Trait '{name}' has invalid value type: {type(trait.value).__name__}")
        
        # Check template variables
        for var_name, var_value in config.template_variables.items():
            if not isinstance(var_name, str) or not var_name:
                errors.append(f"Invalid template variable name: {var_name}")
            
            # Check variable value is of appropriate type
            if not isinstance(var_value, (int, float, str, bool)):
                errors.append(f"Template variable '{var_name}' has invalid value type: {type(var_value).__name__}")
        
        return len(errors) == 0, errors
    
    def check_file_integrity(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Check if a personality configuration file is valid.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        errors = []
        
        # Check file exists
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return False, errors
        
        # Check file is readable
        if not os.access(file_path, os.R_OK):
            errors.append(f"File is not readable: {file_path}")
            return False, errors
        
        # Check file is a valid JSON
        try:
            with open(file_path, "r") as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in file {file_path}: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading file {file_path}: {str(e)}")
            return False, errors
        
        # Validate the personality configuration
        is_valid, validation_errors = self.validate_personality(config_dict)
        errors.extend(validation_errors)
        
        # Check filename matches personality name
        filename = os.path.basename(file_path)
        filename_base = os.path.splitext(filename)[0]
        
        if "name" in config_dict and config_dict["name"] != filename_base:
            errors.append(f"Personality name '{config_dict['name']}' doesn't match filename '{filename_base}'")
        
        return len(errors) == 0, errors
    
    def validate_template_compatibility(
        self, 
        personality: PersonalityConfig, 
        template_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if a personality is compatible with a template.
        
        Args:
            personality: Personality configuration
            template_name: Name of the template to check against
            
        Returns:
            Tuple of (is_compatible, list_of_compatibility_issues)
        """
        issues = []
        
        # Load template
        template = self.template_manager.get_template(template_name)
        if template is None:
            issues.append(f"Template '{template_name}' not found")
            return False, issues
        
        # Get template content
        content = template.get("content")
        if not content:
            issues.append(f"Template '{template_name}' has no content")
            return False, issues
        
        # Check for required variables
        template_vars = set()
        for match in re.finditer(r'\{\{(\w+)\}\}', content):
            template_vars.add(match.group(1))
        
        # Get personality variables
        personality_vars = set(personality.template_variables.keys())
        
        # Add standard trait variables
        personality_vars.add("thinking_style")
        personality_vars.add("reasoning_depth")
        personality_vars.add("uncertainty_approach")
        
        # Add custom trait variables (prefixed with trait_)
        for trait_name in personality.traits.keys():
            personality_vars.add(f"trait_{trait_name}")
        
        # Check for missing variables
        missing_vars = template_vars - personality_vars
        if missing_vars:
            issues.append(f"Personality missing template variables: {', '.join(missing_vars)}")
        
        # Check conditional sections
        conditional_pattern = r'<!-- IF ([\w\.]+) == ([\w\.]+) -->'
        for match in re.finditer(conditional_pattern, content):
            var_name = match.group(1)
            if var_name not in personality_vars and not var_name.startswith("personality."):
                issues.append(f"Personality missing conditional variable: {var_name}")
        
        return len(issues) == 0, issues
    
    def check_consistency(self, personality: PersonalityConfig) -> Tuple[bool, List[str]]:
        """
        Check for internal consistency of a personality.
        
        Args:
            personality: Personality configuration to check
            
        Returns:
            Tuple of (is_consistent, list_of_consistency_issues)
        """
        issues = []
        
        # Check for consistency between thinking style and uncertainty approach
        if personality.thinking_style == ThinkingStyle.ANALYTICAL and personality.uncertainty_approach == UncertaintyApproach.BOLD:
            issues.append("Potential inconsistency: Analytical thinking style with Bold uncertainty approach")
        
        if personality.thinking_style == ThinkingStyle.CREATIVE and personality.uncertainty_approach == UncertaintyApproach.CAUTIOUS:
            issues.append("Potential inconsistency: Creative thinking style with Cautious uncertainty approach")
        
        # Check for consistency between reasoning depth and temperature
        if hasattr(personality, "temperature"):
            if personality.reasoning_depth == ReasoningDepth.EXHAUSTIVE and personality.temperature > 0.8:
                issues.append("Potential inconsistency: Exhaustive reasoning depth with high temperature")
            
            if personality.reasoning_depth == ReasoningDepth.SHALLOW and personality.temperature < 0.3:
                issues.append("Potential inconsistency: Shallow reasoning depth with low temperature")
        
        # Check for trait value consistency
        numeric_traits = {}
        for name, trait in personality.traits.items():
            if isinstance(trait.value, (int, float)):
                numeric_traits[name] = trait.value
        
        # Check if numeric traits are in reasonable ranges
        for name, value in numeric_traits.items():
            if value < 0 or value > 1:
                issues.append(f"Trait '{name}' value {value} is outside the standard 0-1 range")
        
        # Check for semantic contradictions in trait names/values
        contradictions = [
            ("creativity", "conformity"),
            ("speed", "thoroughness"),
            ("risk_taking", "caution"),
            ("innovation", "tradition")
        ]
        
        for trait1, trait2 in contradictions:
            if trait1 in numeric_traits and trait2 in numeric_traits:
                val1, val2 = numeric_traits[trait1], numeric_traits[trait2]
                if val1 > 0.7 and val2 > 0.7:
                    issues.append(f"Potential contradiction: High values for both '{trait1}' ({val1}) and '{trait2}' ({val2})")
        
        return len(issues) == 0, issues

    def validate_personality_directory(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """
        Validate all personality configurations in a directory.
        
        Args:
            directory: Directory containing personality configuration files
            
        Returns:
            Dictionary mapping filenames to validation results
        """
        results = {}
        
        if not os.path.isdir(directory):
            logger.error(f"Not a directory: {directory}")
            return results
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        
        for filename in json_files:
            file_path = os.path.join(directory, filename)
            
            # Check file integrity
            is_valid, errors = self.check_file_integrity(file_path)
            
            if is_valid:
                # Load personality for further checks
                try:
                    with open(file_path, "r") as f:
                        config_dict = json.load(f)
                    
                    personality = PersonalityConfig.from_dict(config_dict)
                    
                    # Check internal consistency
                    is_consistent, consistency_issues = self.check_consistency(personality)
                    
                    # Check template compatibility with a few common templates
                    template_compatibility = {}
                    for template_name in self.template_manager.discover_templates():
                        is_compatible, issues = self.validate_template_compatibility(personality, template_name)
                        template_compatibility[template_name] = {
                            "compatible": is_compatible,
                            "issues": issues
                        }
                    
                    results[filename] = {
                        "valid": True,
                        "consistency": {
                            "consistent": is_consistent,
                            "issues": consistency_issues
                        },
                        "template_compatibility": template_compatibility
                    }
                except Exception as e:
                    results[filename] = {
                        "valid": False,
                        "errors": [f"Error performing additional checks: {str(e)}"]
                    }
            else:
                results[filename] = {
                    "valid": False,
                    "errors": errors
                }
        
        return results
    
    def generate_validation_report(
        self, 
        validation_results: Dict[str, Dict[str, Any]],
        include_passes: bool = False
    ) -> str:
        """
        Generate a readable validation report from validation results.
        
        Args:
            validation_results: Results from validate_personality_directory
            include_passes: Whether to include passing files in the report
            
        Returns:
            Formatted validation report
        """
        report = "# Personality Validation Report\n\n"
        
        # Count valid and invalid files
        valid_count = sum(1 for result in validation_results.values() if result.get("valid", False))
        invalid_count = len(validation_results) - valid_count
        
        report += f"## Summary\n"
        report += f"- Total files: {len(validation_results)}\n"
        report += f"- Valid: {valid_count}\n"
        report += f"- Invalid: {invalid_count}\n\n"
        
        # Report invalid files
        if invalid_count > 0:
            report += f"## Invalid Files\n"
            for filename, result in validation_results.items():
                if not result.get("valid", False):
                    report += f"### {filename}\n"
                    for error in result.get("errors", []):
                        report += f"- {error}\n"
                    report += "\n"
        
        # Report files with consistency issues
        consistency_issues_count = 0
        for result in validation_results.values():
            if result.get("valid", False) and not result.get("consistency", {}).get("consistent", True):
                consistency_issues_count += 1
        
        if consistency_issues_count > 0:
            report += f"## Consistency Issues\n"
            for filename, result in validation_results.items():
                if result.get("valid", False):
                    consistency = result.get("consistency", {})
                    if not consistency.get("consistent", True):
                        report += f"### {filename}\n"
                        for issue in consistency.get("issues", []):
                            report += f"- {issue}\n"
                        report += "\n"
        
        # Report template compatibility issues
        template_issues = {}
        for filename, result in validation_results.items():
            if result.get("valid", False):
                compat = result.get("template_compatibility", {})
                for template_name, template_result in compat.items():
                    if not template_result.get("compatible", True):
                        if template_name not in template_issues:
                            template_issues[template_name] = []
                        template_issues[template_name].append((filename, template_result.get("issues", [])))
        
        if template_issues:
            report += f"## Template Compatibility Issues\n"
            for template_name, files in template_issues.items():
                report += f"### Template: {template_name}\n"
                for filename, issues in files:
                    report += f"#### {filename}\n"
                    for issue in issues:
                        report += f"- {issue}\n"
                    report += "\n"
        
        # Report valid files if requested
        if include_passes and valid_count > 0:
            report += f"## Valid Files\n"
            for filename, result in validation_results.items():
                if result.get("valid", False) and result.get("consistency", {}).get("consistent", True):
                    report += f"- {filename}\n"
            report += "\n"
        
        return report


def validate_personality_file(file_path: str) -> bool:
    """
    Validate a personality configuration file.
    
    Args:
        file_path: Path to the personality configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = PersonalityValidator()
    is_valid, errors = validator.check_file_integrity(file_path)
    
    if not is_valid:
        for error in errors:
            logger.error(f"Validation error in {file_path}: {error}")
    
    return is_valid 