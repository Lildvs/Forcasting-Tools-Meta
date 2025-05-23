"""
Personality System Health Check Utilities

This module provides utilities to monitor the health and functionality of the
personality management system, detect issues, and provide diagnostics.
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.template_manager import TemplateManager
from forecasting_tools.personality_management.cache import PersonalityCache
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer
from forecasting_tools.personality_management.validators import PersonalityValidator
from forecasting_tools.personality_management.feature_flags import get_feature_flags

logger = logging.getLogger(__name__)


class PersonalitySystemHealthCheck:
    """
    Health check utilities for the personality management system.
    
    This class provides methods to verify that all components of the
    personality system are functioning correctly, detect issues, 
    and generate health reports.
    """
    
    def __init__(self, check_level: str = "standard"):
        """
        Initialize the health check.
        
        Args:
            check_level: Level of health check ("minimal", "standard", or "comprehensive")
        """
        self.check_level = check_level
        self.manager = PersonalityManager()
        self.template_manager = TemplateManager()
        self.cache = PersonalityCache()
        self.optimizer = PromptOptimizer()
        self.validator = PersonalityValidator()
        self.feature_flags = get_feature_flags()
        
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "check_level": check_level,
            "overall_status": "unknown",
            "components": {},
            "warnings": [],
            "errors": []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks based on the specified check level.
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Core functionality checks
            self._check_feature_flags()
            self._check_personality_loading()
            self._check_template_loading()
            
            # Additional checks based on level
            if self.check_level in ("standard", "comprehensive"):
                self._check_personality_validation()
                self._check_template_rendering()
                self._check_cache_functionality()
            
            if self.check_level == "comprehensive":
                self._check_prompt_generation()
                self._check_performance()
                self._check_compatibility()
            
            # Set overall status
            if len(self.results["errors"]) > 0:
                self.results["overall_status"] = "error"
            elif len(self.results["warnings"]) > 0:
                self.results["overall_status"] = "warning"
            else:
                self.results["overall_status"] = "healthy"
                
            # Add environment information
            self._add_environment_info()
            
        except Exception as e:
            self.results["overall_status"] = "error"
            self.results["errors"].append(f"Health check error: {str(e)}")
            logger.error(f"Health check failed: {str(e)}\n{traceback.format_exc()}")
        
        # Add execution time
        self.results["execution_time"] = f"{(datetime.now() - datetime.fromisoformat(self.results['timestamp'])).total_seconds():.2f} seconds"
        
        return self.results
    
    def _check_feature_flags(self) -> None:
        """Check feature flag functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            # Check if main feature flag is available
            enabled = self.feature_flags.is_enabled("personality_system_enabled")
            component_result["details"]["system_enabled"] = enabled
            
            # Get all flags
            all_flags = self.feature_flags.get_all_flags()
            component_result["details"]["flags_count"] = len(all_flags)
            
            # Test flag update functionality
            original_value = all_flags.get("verbose_logging", False)
            self.feature_flags.update_flag("verbose_logging", not original_value)
            updated_value = self.feature_flags.get_all_flags().get("verbose_logging", False)
            self.feature_flags.update_flag("verbose_logging", original_value)  # Restore original
            
            if updated_value == original_value:
                component_result["status"] = "warning"
                self.results["warnings"].append("Feature flag update functionality not working properly")
            
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Feature flag check failed: {str(e)}")
        
        self.results["components"]["feature_flags"] = component_result
    
    def _check_personality_loading(self) -> None:
        """Check personality loading functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            # List available personalities
            personalities = self.manager.list_available_personalities()
            component_result["details"]["available_personalities"] = len(personalities)
            
            if len(personalities) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("No personalities found")
            else:
                # Try to load the first personality
                test_personality_name = personalities[0]
                personality = self.manager.load_personality(test_personality_name)
                
                if not personality:
                    component_result["status"] = "error"
                    self.results["errors"].append(f"Failed to load personality: {test_personality_name}")
                else:
                    component_result["details"]["test_personality"] = test_personality_name
                    component_result["details"]["load_success"] = True
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Personality loading check failed: {str(e)}")
        
        self.results["components"]["personality_loading"] = component_result
    
    def _check_template_loading(self) -> None:
        """Check template loading functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            # Check template directories
            component_result["details"]["template_directories"] = len(self.template_manager._template_dirs)
            
            # Discover templates
            templates = self.template_manager.discover_templates()
            component_result["details"]["available_templates"] = len(templates)
            
            if len(templates) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("No templates found")
            else:
                # Try to load the first template
                test_template_name = templates[0]
                template = self.template_manager.get_template(test_template_name)
                
                if not template:
                    component_result["status"] = "error"
                    self.results["errors"].append(f"Failed to load template: {test_template_name}")
                else:
                    component_result["details"]["test_template"] = test_template_name
                    component_result["details"]["load_success"] = True
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Template loading check failed: {str(e)}")
        
        self.results["components"]["template_loading"] = component_result
    
    def _check_personality_validation(self) -> None:
        """Check personality validation functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            personalities = self.manager.list_available_personalities()
            
            if len(personalities) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("No personalities to validate")
            else:
                # Validate a subset of personalities
                test_personalities = personalities[:min(5, len(personalities))]
                validation_results = []
                
                for name in test_personalities:
                    personality = self.manager.load_personality(name)
                    if personality:
                        is_valid, errors = self.validator.validate_personality(personality)
                        validation_results.append({
                            "name": name,
                            "valid": is_valid,
                            "errors": errors
                        })
                
                component_result["details"]["validation_results"] = validation_results
                
                # Check if any validations failed
                invalid_personalities = [r for r in validation_results if not r["valid"]]
                if invalid_personalities:
                    component_result["status"] = "warning"
                    for p in invalid_personalities:
                        self.results["warnings"].append(f"Personality validation failed for {p['name']}: {p['errors']}")
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Personality validation check failed: {str(e)}")
        
        self.results["components"]["personality_validation"] = component_result
    
    def _check_template_rendering(self) -> None:
        """Check template rendering functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            templates = self.template_manager.discover_templates()
            
            if len(templates) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("No templates to render")
            else:
                # Test rendering a template with variables
                test_template_name = next((t for t in templates if "forecast" in t.lower()), templates[0])
                test_variables = {
                    "question": "Test question for health check",
                    "thinking_style": "analytical",
                    "uncertainty_approach": "balanced",
                    "reasoning_depth": "moderate"
                }
                
                rendered = self.template_manager.render_template(test_template_name, test_variables)
                
                if not rendered:
                    component_result["status"] = "error"
                    self.results["errors"].append(f"Failed to render template: {test_template_name}")
                else:
                    component_result["details"]["test_template"] = test_template_name
                    component_result["details"]["render_success"] = True
                    component_result["details"]["rendered_length"] = len(rendered)
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Template rendering check failed: {str(e)}")
        
        self.results["components"]["template_rendering"] = component_result
    
    def _check_cache_functionality(self) -> None:
        """Check cache functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            # Get cache stats before tests
            pre_stats = self.cache.get_stats()
            component_result["details"]["initial_cache_size"] = pre_stats.get("size", 0)
            
            # Clear cache to ensure clean test
            self.cache.invalidate_all()
            
            # Load a personality to cache
            personalities = self.manager.list_available_personalities()
            if len(personalities) > 0:
                test_personality_name = personalities[0]
                
                # First load (should cache)
                personality1 = self.manager.load_personality(test_personality_name)
                
                # Second load (should use cache)
                start_time = time.time()
                personality2 = self.manager.load_personality(test_personality_name)
                cached_load_time = time.time() - start_time
                
                # Get cache stats after test
                post_stats = self.cache.get_stats()
                
                component_result["details"]["final_cache_size"] = post_stats.get("size", 0)
                component_result["details"]["cache_hits"] = post_stats.get("hits", 0)
                component_result["details"]["cached_load_time"] = cached_load_time
                
                # Verify the cache is working
                if post_stats.get("hits", 0) <= 0:
                    component_result["status"] = "warning"
                    self.results["warnings"].append("Cache may not be working properly (no hits recorded)")
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Cache functionality check failed: {str(e)}")
        
        self.results["components"]["cache_functionality"] = component_result
    
    def _check_prompt_generation(self) -> None:
        """Check prompt generation functionality."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            personalities = self.manager.list_available_personalities()
            templates = self.template_manager.discover_templates()
            
            if len(personalities) == 0 or len(templates) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("Cannot test prompt generation without personalities and templates")
            else:
                # Find a suitable template for testing
                test_template_name = next((t for t in templates if "forecast" in t.lower()), templates[0])
                test_personality_name = next((p for p in personalities if p in ["analytical", "balanced"]), personalities[0])
                
                # Generate a prompt
                prompt, metadata = self.optimizer.optimize_prompt_pipeline(
                    personality_name=test_personality_name,
                    template_name=test_template_name,
                    variables={"question": "Test question for health check"}
                )
                
                if not prompt or len(prompt) < 10:
                    component_result["status"] = "error"
                    self.results["errors"].append("Prompt generation failed or produced invalid output")
                else:
                    component_result["details"]["test_personality"] = test_personality_name
                    component_result["details"]["test_template"] = test_template_name
                    component_result["details"]["prompt_length"] = len(prompt)
                    component_result["details"]["estimated_tokens"] = metadata.get("estimated_tokens", 0)
                
                # Test prompt cache functionality
                self.optimizer.clear_cache()  # Clear cache first
                
                # Generate same prompt again
                start_time = time.time()
                self.optimizer.optimize_prompt_pipeline(
                    personality_name=test_personality_name,
                    template_name=test_template_name,
                    variables={"question": "Test question for health check"}
                )
                first_gen_time = time.time() - start_time
                
                # Generate again (should use cache)
                start_time = time.time()
                self.optimizer.optimize_prompt_pipeline(
                    personality_name=test_personality_name,
                    template_name=test_template_name,
                    variables={"question": "Test question for health check"}
                )
                second_gen_time = time.time() - start_time
                
                component_result["details"]["first_generation_time"] = first_gen_time
                component_result["details"]["cached_generation_time"] = second_gen_time
                
                # Check if caching is working (second should be faster)
                if second_gen_time >= first_gen_time:
                    component_result["status"] = "warning"
                    self.results["warnings"].append("Prompt caching may not be working properly")
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Prompt generation check failed: {str(e)}")
        
        self.results["components"]["prompt_generation"] = component_result
    
    def _check_performance(self) -> None:
        """Check performance metrics."""
        component_result = {
            "status": "healthy",
            "details": {
                "benchmarks": {}
            }
        }
        
        try:
            personalities = self.manager.list_available_personalities()
            templates = self.template_manager.discover_templates()
            
            if len(personalities) == 0 or len(templates) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("Cannot test performance without personalities and templates")
            else:
                # Performance test parameters
                test_template_name = next((t for t in templates if "forecast" in t.lower()), templates[0])
                test_personality_name = next((p for p in personalities if p in ["analytical", "balanced"]), personalities[0])
                test_question = "Will AI significantly impact global labor markets in the next decade?"
                
                # Clear caches first
                self.cache.invalidate_all()
                self.optimizer.clear_cache()
                
                # Benchmark: Personality loading
                start_time = time.time()
                for _ in range(5):
                    self.manager.load_personality(test_personality_name)
                personality_load_time = (time.time() - start_time) / 5
                
                # Benchmark: Template rendering
                test_variables = {
                    "question": test_question,
                    "thinking_style": "analytical",
                    "uncertainty_approach": "balanced",
                    "reasoning_depth": "moderate"
                }
                
                start_time = time.time()
                for _ in range(5):
                    self.template_manager.render_template(test_template_name, test_variables)
                template_render_time = (time.time() - start_time) / 5
                
                # Benchmark: Prompt generation
                start_time = time.time()
                self.optimizer.optimize_prompt_pipeline(
                    personality_name=test_personality_name,
                    template_name=test_template_name,
                    variables={"question": test_question}
                )
                prompt_gen_time = time.time() - start_time
                
                # Store benchmark results
                component_result["details"]["benchmarks"] = {
                    "personality_load_time": personality_load_time,
                    "template_render_time": template_render_time,
                    "prompt_generation_time": prompt_gen_time
                }
                
                # Check for performance issues
                if personality_load_time > 0.1:
                    component_result["status"] = "warning"
                    self.results["warnings"].append(f"Slow personality loading: {personality_load_time:.4f} seconds")
                
                if template_render_time > 0.05:
                    component_result["status"] = "warning"
                    self.results["warnings"].append(f"Slow template rendering: {template_render_time:.4f} seconds")
                
                if prompt_gen_time > 0.5:
                    component_result["status"] = "warning"
                    self.results["warnings"].append(f"Slow prompt generation: {prompt_gen_time:.4f} seconds")
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Performance check failed: {str(e)}")
        
        self.results["components"]["performance"] = component_result
    
    def _check_compatibility(self) -> None:
        """Check compatibility between personalities and templates."""
        component_result = {
            "status": "healthy",
            "details": {}
        }
        
        try:
            personalities = self.manager.list_available_personalities()
            templates = self.template_manager.discover_templates()
            
            if len(personalities) == 0 or len(templates) == 0:
                component_result["status"] = "warning"
                self.results["warnings"].append("Cannot test compatibility without personalities and templates")
            else:
                # Test a subset of personalities and templates
                test_personalities = personalities[:min(3, len(personalities))]
                test_templates = templates[:min(3, len(templates))]
                
                compatibility_results = []
                
                for p_name in test_personalities:
                    personality = self.manager.load_personality(p_name)
                    if not personality:
                        continue
                        
                    for t_name in test_templates:
                        is_compatible, issues = self.validator.validate_template_compatibility(
                            personality, t_name
                        )
                        
                        compatibility_results.append({
                            "personality": p_name,
                            "template": t_name,
                            "compatible": is_compatible,
                            "issues": issues
                        })
                
                component_result["details"]["compatibility_results"] = compatibility_results
                
                # Check for compatibility issues
                incompatible_pairs = [r for r in compatibility_results if not r["compatible"]]
                if incompatible_pairs:
                    component_result["status"] = "warning"
                    for pair in incompatible_pairs:
                        self.results["warnings"].append(
                            f"Compatibility issue between {pair['personality']} and {pair['template']}: {pair['issues']}"
                        )
        
        except Exception as e:
            component_result["status"] = "error"
            component_result["error"] = str(e)
            self.results["errors"].append(f"Compatibility check failed: {str(e)}")
        
        self.results["components"]["compatibility"] = component_result
    
    def _add_environment_info(self) -> None:
        """Add environment information to the results."""
        environment = {
            "python_version": None,
            "os_info": None,
            "personality_dirs": None,
            "template_dirs": None
        }
        
        try:
            import platform
            import sys
            
            environment["python_version"] = sys.version
            environment["os_info"] = platform.platform()
            
            # Get personality and template directories
            personality_dirs = os.environ.get("PERSONALITY_DIRS", "").split(os.pathsep)
            template_dirs = self.template_manager._template_dirs
            
            environment["personality_dirs"] = personality_dirs
            environment["template_dirs"] = template_dirs
            
        except Exception as e:
            environment["error"] = str(e)
        
        self.results["environment"] = environment
    
    def save_report(self, file_path: str) -> bool:
        """
        Save the health check report to a file.
        
        Args:
            file_path: Path to save the report
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save report to file
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Health check report saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save health check report: {str(e)}")
            return False


def check_system_health(
    check_level: str = "standard",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a health check on the personality system.
    
    Args:
        check_level: Level of health check ("minimal", "standard", or "comprehensive")
        output_file: Optional path to save the report
        
    Returns:
        Health check results
    """
    checker = PersonalitySystemHealthCheck(check_level=check_level)
    results = checker.run_all_checks()
    
    if output_file:
        checker.save_report(output_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run health checks on the personality system")
    parser.add_argument(
        "--level", 
        choices=["minimal", "standard", "comprehensive"],
        default="standard",
        help="Level of health check to perform"
    )
    parser.add_argument(
        "--output",
        help="Path to save the health check report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print(f"Running {args.level} health check on personality system...")
    
    results = check_system_health(args.level, args.output)
    
    # Print summary
    print(f"\nHealth Check Summary: {results['overall_status'].upper()}")
    print(f"Components checked: {len(results['components'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"- {warning}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"- {error}")
    
    if args.output:
        print(f"\nDetailed report saved to: {args.output}")
    else:
        print("\nRun with --output PATH to save detailed report")
        
    # Exit with appropriate status code
    if results['overall_status'] == "error":
        import sys
        sys.exit(1) 