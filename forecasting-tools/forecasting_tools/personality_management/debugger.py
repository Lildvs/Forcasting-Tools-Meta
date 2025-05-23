"""
Personality Debugger Tools

This module provides tools for diagnosing and resolving issues with
personality configurations and their integration with forecasting bots.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable

from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.template_manager import TemplateManager
from forecasting_tools.personality_management.validators import PersonalityValidator
from forecasting_tools.personality_management.cache import PersonalityCache
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class PersonalityDebugger:
    """
    Debugger for personality configurations and template integrations.
    
    This class provides tools to diagnose issues with personality configurations,
    template rendering, and integration with forecasting bots.
    """
    
    def __init__(self):
        """Initialize the debugger."""
        self.validator = PersonalityValidator()
        self.template_manager = TemplateManager()
        self.cache = PersonalityCache()
        self.prompt_optimizer = PromptOptimizer()
        self.debug_logs: List[Dict[str, Any]] = []
        self.log_level = logging.INFO
    
    def set_log_level(self, level: int):
        """
        Set the log level for the debugger.
        
        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        self.log_level = level
    
    def _log(self, level: int, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Add an entry to the debug log.
        
        Args:
            level: Log level
            message: Log message
            data: Additional data for the log entry
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {}
        }
        
        self.debug_logs.append(entry)
        
        if level >= self.log_level:
            if data:
                logger.log(level, f"{message} - {json.dumps(data, default=str)}")
            else:
                logger.log(level, message)
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log(logging.DEBUG, message, data)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log(logging.INFO, message, data)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self._log(logging.WARNING, message, data)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log an error message."""
        self._log(logging.ERROR, message, data)
    
    def get_logs(self, level: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get filtered debug logs.
        
        Args:
            level: Minimum log level to include (None for all logs)
            
        Returns:
            Filtered list of log entries
        """
        if level is None:
            return self.debug_logs
        
        return [entry for entry in self.debug_logs if entry["level"] >= level]
    
    def clear_logs(self):
        """Clear all debug logs."""
        self.debug_logs = []
    
    def export_logs(self, file_path: str) -> bool:
        """
        Export debug logs to a JSON file.
        
        Args:
            file_path: Path to save the logs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "logs": self.debug_logs
                }, f, indent=2, default=str)
            return True
        except (IOError, ValueError) as e:
            logger.error(f"Failed to export debug logs: {str(e)}")
            return False
    
    def diagnose_personality(self, personality: Union[PersonalityConfig, str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Diagnose issues with a personality configuration.
        
        Args:
            personality: Personality configuration object, name, or dictionary
            
        Returns:
            Diagnostic results
        """
        self.info(f"Diagnosing personality: {personality if isinstance(personality, str) else ''}")
        
        # Convert to PersonalityConfig object
        config = None
        try:
            if isinstance(personality, str):
                # Assume it's a personality name or file path
                if os.path.exists(personality):
                    # It's a file path
                    with open(personality, "r") as f:
                        config_dict = json.load(f)
                    
                    config = PersonalityConfig.from_dict(config_dict)
                    self.debug(f"Loaded personality from file: {personality}")
                else:
                    # Assume it's a personality name
                    try:
                        from forecasting_tools.personality_management import PersonalityManager
                        manager = PersonalityManager()
                        config = manager.load_personality(personality)
                        self.debug(f"Loaded personality from manager: {personality}")
                    except Exception as e:
                        self.error(f"Failed to load personality by name: {str(e)}")
                        return {
                            "success": False,
                            "error": f"Failed to load personality: {str(e)}",
                            "details": traceback.format_exc()
                        }
            elif isinstance(personality, dict):
                config = PersonalityConfig.from_dict(personality)
                self.debug("Loaded personality from dictionary")
            else:
                config = personality
                self.debug("Using provided PersonalityConfig object")
        except Exception as e:
            self.error(f"Failed to load personality configuration: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to load personality configuration: {str(e)}",
                "details": traceback.format_exc()
            }
        
        # Initialize results
        results = {
            "success": True,
            "personality": {
                "name": config.name,
                "thinking_style": config.thinking_style,
                "uncertainty_approach": config.uncertainty_approach,
                "reasoning_depth": config.reasoning_depth,
                "traits": {k: v.value for k, v in config.traits.items()},
                "template_variables": dict(config.template_variables)
            },
            "validation": {},
            "template_compatibility": {},
            "cache_status": None,
            "issues": [],
            "warnings": []
        }
        
        # Run validation
        is_valid, validation_errors = self.validator.validate_personality(config)
        results["validation"] = {
            "valid": is_valid,
            "errors": validation_errors
        }
        
        if not is_valid:
            results["success"] = False
            results["issues"].extend([f"Validation error: {e}" for e in validation_errors])
            self.warning("Personality validation failed", {"errors": validation_errors})
        
        # Check consistency
        is_consistent, consistency_issues = self.validator.check_consistency(config)
        results["consistency"] = {
            "consistent": is_consistent,
            "issues": consistency_issues
        }
        
        if not is_consistent:
            results["warnings"].extend([f"Consistency issue: {i}" for i in consistency_issues])
            self.warning("Personality consistency issues found", {"issues": consistency_issues})
        
        # Check template compatibility
        template_compatibility = {}
        for template_name in self.template_manager.discover_templates():
            is_compatible, issues = self.validator.validate_template_compatibility(config, template_name)
            template_compatibility[template_name] = {
                "compatible": is_compatible,
                "issues": issues
            }
            
            if not is_compatible:
                results["warnings"].append(f"Template compatibility issue with '{template_name}': {issues[0]}")
                self.warning(f"Template compatibility issue with '{template_name}'", {"issues": issues})
        
        results["template_compatibility"] = template_compatibility
        
        # Check cache status
        cache_key = self.cache.get_cache_key(config.name)
        cache_hit = self.cache.has_personality(config.name)
        results["cache_status"] = {
            "cache_key": cache_key,
            "in_cache": cache_hit
        }
        
        self.info("Personality diagnosis complete", {"results": results})
        return results
    
    def test_template_rendering(
        self, 
        personality_name: str, 
        template_name: str, 
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Test template rendering with a personality.
        
        Args:
            personality_name: Name of the personality
            template_name: Name of the template
            variables: Additional template variables
            
        Returns:
            Test results with rendered content
        """
        self.info(f"Testing template rendering: {template_name} with {personality_name}")
        
        results = {
            "success": False,
            "personality_name": personality_name,
            "template_name": template_name,
            "variables": variables or {},
            "rendered_content": None,
            "error": None,
            "template_info": None,
            "warnings": []
        }
        
        try:
            # Load template
            template = self.template_manager.get_template(template_name)
            if template is None:
                error = f"Template '{template_name}' not found"
                self.error(error)
                results["error"] = error
                return results
            
            results["template_info"] = {
                "content_length": len(template.get("content", "")),
                "variables": template.get("variables", {})
            }
            
            # Load personality
            try:
                from forecasting_tools.personality_management import PersonalityManager
                manager = PersonalityManager()
                personality = manager.load_personality(personality_name)
            except Exception as e:
                error = f"Failed to load personality '{personality_name}': {str(e)}"
                self.error(error)
                results["error"] = error
                return results
            
            # Check compatibility
            is_compatible, issues = self.validator.validate_template_compatibility(personality, template_name)
            results["compatibility"] = {
                "compatible": is_compatible,
                "issues": issues
            }
            
            if not is_compatible:
                results["warnings"].extend([f"Compatibility issue: {i}" for i in issues])
                self.warning("Template compatibility issues found", {"issues": issues})
            
            # Merge variables
            merged_variables = {}
            
            # Add template default variables
            if "variables" in template:
                merged_variables.update(template["variables"])
            
            # Add personality template variables
            merged_variables.update(personality.template_variables)
            
            # Add basic personality traits
            merged_variables.update({
                "thinking_style": personality.thinking_style,
                "uncertainty_approach": personality.uncertainty_approach,
                "reasoning_depth": personality.reasoning_depth
            })
            
            # Add custom traits
            for trait_name, trait in personality.traits.items():
                merged_variables[f"trait_{trait_name}"] = trait.value
            
            # Add custom variables
            if variables:
                merged_variables.update(variables)
            
            # Render template
            rendered = self.template_manager.render_template(template_name, merged_variables)
            if rendered is None:
                error = "Template rendering failed"
                self.error(error)
                results["error"] = error
                return results
            
            results["rendered_content"] = rendered
            results["applied_variables"] = merged_variables
            results["success"] = True
            
            self.info("Template rendering successful", {"content_length": len(rendered)})
            
        except Exception as e:
            error = f"Template rendering error: {str(e)}"
            self.error(error, {"traceback": traceback.format_exc()})
            results["error"] = error
            
        return results
    
    def analyze_prompt_pipeline(
        self, 
        personality_name: str, 
        template_name: str, 
        variables: Optional[Dict[str, Any]] = None, 
        context_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the full prompt generation pipeline.
        
        Args:
            personality_name: Name of the personality
            template_name: Name of the template
            variables: Additional template variables
            context_size: Context size limit for optimization
            
        Returns:
            Analysis results with timing and content
        """
        self.info(f"Analyzing prompt pipeline: {template_name} with {personality_name}")
        
        results = {
            "success": False,
            "personality_name": personality_name,
            "template_name": template_name,
            "variables": variables or {},
            "context_size": context_size,
            "timing": {},
            "output": None,
            "metadata": None,
            "error": None
        }
        
        try:
            import time
            
            # Measure overall time
            start_time = time.time()
            
            # Measure personality loading time
            personality_start = time.time()
            try:
                from forecasting_tools.personality_management import PersonalityManager
                manager = PersonalityManager()
                personality = manager.load_personality(personality_name)
                if personality is None:
                    error = f"Personality '{personality_name}' not found"
                    self.error(error)
                    results["error"] = error
                    return results
            except Exception as e:
                error = f"Failed to load personality '{personality_name}': {str(e)}"
                self.error(error)
                results["error"] = error
                return results
            
            personality_time = time.time() - personality_start
            results["timing"]["personality_loading"] = personality_time
            
            # Measure template loading time
            template_start = time.time()
            template = self.template_manager.get_template(template_name)
            if template is None:
                error = f"Template '{template_name}' not found"
                self.error(error)
                results["error"] = error
                return results
            template_time = time.time() - template_start
            results["timing"]["template_loading"] = template_time
            
            # Process the prompt
            pipeline_start = time.time()
            prompt, metadata = self.prompt_optimizer.optimize_prompt_pipeline(
                personality_name=personality_name,
                template_name=template_name,
                variables=variables or {},
                context_size=context_size
            )
            pipeline_time = time.time() - pipeline_start
            results["timing"]["pipeline_processing"] = pipeline_time
            
            total_time = time.time() - start_time
            results["timing"]["total"] = total_time
            
            # Record output
            results["output"] = prompt
            results["metadata"] = metadata
            results["success"] = True
            
            # Add timing breakdown percentage
            results["timing"]["breakdown"] = {
                "personality_loading_pct": (personality_time / total_time) * 100 if total_time > 0 else 0,
                "template_loading_pct": (template_time / total_time) * 100 if total_time > 0 else 0,
                "pipeline_processing_pct": (pipeline_time / total_time) * 100 if total_time > 0 else 0
            }
            
            self.info("Prompt pipeline analysis complete", {
                "timing": results["timing"],
                "output_length": len(prompt) if prompt else 0,
                "estimated_tokens": metadata.get("estimated_tokens", 0)
            })
            
        except Exception as e:
            error = f"Prompt pipeline analysis error: {str(e)}"
            self.error(error, {"traceback": traceback.format_exc()})
            results["error"] = error
            
        return results
    
    def simulate_bot_integration(
        self, 
        personality_name: str, 
        template_name: str, 
        variables: Optional[Dict[str, Any]] = None,
        mock_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate integration with a forecasting bot.
        
        Args:
            personality_name: Name of the personality
            template_name: Name of the template
            variables: Additional template variables
            mock_response: Optional mock response (if None, will simulate)
            
        Returns:
            Simulation results
        """
        self.info(f"Simulating bot integration: {template_name} with {personality_name}")
        
        results = {
            "success": False,
            "personality_name": personality_name,
            "template_name": template_name,
            "variables": variables or {},
            "prompt": None,
            "response": None,
            "error": None,
            "timing": {}
        }
        
        try:
            import time
            
            # Generate prompt
            prompt_start = time.time()
            prompt, metadata = self.prompt_optimizer.optimize_prompt_pipeline(
                personality_name=personality_name,
                template_name=template_name,
                variables=variables or {}
            )
            prompt_time = time.time() - prompt_start
            results["timing"]["prompt_generation"] = prompt_time
            
            if prompt is None:
                error = "Failed to generate prompt"
                self.error(error)
                results["error"] = error
                return results
            
            results["prompt"] = prompt
            results["prompt_metadata"] = metadata
            
            # Generate mock response or use provided one
            response_start = time.time()
            if mock_response is not None:
                response = mock_response
            else:
                # Simple mock response generator
                response = self._generate_mock_bot_response(prompt, personality_name)
            
            response_time = time.time() - response_start
            results["timing"]["response_generation"] = response_time
            
            results["response"] = response
            results["success"] = True
            
            self.info("Bot integration simulation complete", {
                "prompt_length": len(prompt),
                "response_length": len(response)
            })
            
        except Exception as e:
            error = f"Bot integration simulation error: {str(e)}"
            self.error(error, {"traceback": traceback.format_exc()})
            results["error"] = error
            
        return results
    
    def _generate_mock_bot_response(self, prompt: str, personality_name: str) -> str:
        """
        Generate a mock bot response for simulation.
        
        Args:
            prompt: Generated prompt
            personality_name: Name of the personality
            
        Returns:
            Mock response
        """
        # Get personality traits to influence response
        try:
            from forecasting_tools.personality_management import PersonalityManager
            manager = PersonalityManager()
            personality = manager.load_personality(personality_name)
            
            thinking_style = personality.thinking_style
            uncertainty_approach = personality.uncertainty_approach
            reasoning_depth = personality.reasoning_depth
            
            # Generate a mock response based on personality traits
            response_parts = []
            
            # Intro based on thinking style
            if thinking_style == "analytical":
                response_parts.append("After careful analysis of the available data, I've determined that...")
            elif thinking_style == "creative":
                response_parts.append("Considering multiple unconventional perspectives, I believe that...")
            elif thinking_style == "bayesian":
                response_parts.append("Based on my prior beliefs updated with the new evidence, I estimate that...")
            else:  # balanced
                response_parts.append("Taking a balanced approach to this question, I think that...")
            
            # Body based on reasoning depth
            if reasoning_depth == "shallow":
                response_parts.append("The key factors suggest a straightforward conclusion.")
            elif reasoning_depth == "moderate":
                response_parts.append("Several important factors contribute to this outcome. First, consider the historical precedents. Second, current trends indicate a pattern. Third, expert consensus provides additional support.")
            elif reasoning_depth == "deep":
                response_parts.append("This question requires deeper analysis. I've examined historical data, current trends, expert opinions, counterarguments, and potential confounding factors. The interplay between these elements suggests a complex but discernible pattern.")
            elif reasoning_depth == "exhaustive":
                response_parts.append("After exhaustive investigation from multiple angles, I've constructed a comprehensive model of the situation. The historical context reveals important patterns, while current data indicates evolving trends. Expert opinions vary but converge around certain key insights. Alternative hypotheses have been systematically evaluated and either incorporated or rejected based on their explanatory power.")
            
            # Conclusion based on uncertainty approach
            if uncertainty_approach == "cautious":
                response_parts.append("Given the significant uncertainties involved, I'd estimate a probability of 60% with low confidence.")
            elif uncertainty_approach == "bold":
                response_parts.append("Despite some uncertainties, the evidence strongly points to a probability of 80% with high confidence.")
            else:  # balanced
                response_parts.append("Balancing the evidence and uncertainties, I estimate a probability of 70% with moderate confidence.")
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            self.warning(f"Error generating personalized mock response: {str(e)}")
            # Fallback generic response
            return "This is a mock response for simulation purposes. In a real scenario, the forecasting bot would generate content influenced by the personality traits."
    
    def analyze_cache_performance(self, clear_cache: bool = False) -> Dict[str, Any]:
        """
        Analyze the performance of the cache.
        
        Args:
            clear_cache: Whether to clear the cache before analysis
            
        Returns:
            Cache performance metrics
        """
        self.info("Analyzing cache performance")
        
        if clear_cache:
            self.cache.invalidate_all()
            self.template_manager.invalidate_all_templates()
            self.prompt_optimizer.clear_cache()
            self.info("Cleared all caches")
        
        # Get cache stats
        personality_cache_stats = self.cache.get_stats()
        prompt_cache_stats = self.prompt_optimizer.get_cache_stats()
        
        # Generate results
        results = {
            "personality_cache": personality_cache_stats,
            "prompt_cache": prompt_cache_stats,
            "template_cache": {
                "cached_templates": len(self.template_manager._template_cache),
                "directories": len(self.template_manager._template_dirs)
            }
        }
        
        self.info("Cache performance analysis complete", results)
        return results
    
    def generate_debug_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive debug report.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            Report data
        """
        self.info("Generating debug report")
        
        # Collect system information
        import platform
        import sys
        
        system_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine()
        }
        
        # Collect cache information
        cache_info = self.analyze_cache_performance(clear_cache=False)
        
        # Collect template information
        templates = self.template_manager.discover_templates()
        template_info = {
            "count": len(templates),
            "names": templates,
            "directories": self.template_manager._template_dirs
        }
        
        # Create the report
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_info": system_info,
            "cache_info": cache_info,
            "template_info": template_info,
            "logs": self.get_logs()
        }
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                self.info(f"Debug report saved to {output_file}")
            except (IOError, ValueError) as e:
                self.error(f"Failed to save debug report: {str(e)}")
        
        return report


# Singleton debugger instance for convenience
_debugger = None

def get_debugger() -> PersonalityDebugger:
    """Get the singleton debugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = PersonalityDebugger()
    return _debugger 