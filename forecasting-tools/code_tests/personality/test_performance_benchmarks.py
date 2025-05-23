"""
Performance benchmarks for personality configurations.

This module provides benchmarks to evaluate the performance impact of
different personality configurations and identify optimization opportunities.
"""

import unittest
import os
import tempfile
import json
import time
import random
from typing import Dict, Any, List, Optional

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    ThinkingStyle,
    UncertaintyApproach,
    ReasoningDepth
)
from forecasting_tools.personality_management.cache import PersonalityCache
from forecasting_tools.personality_management.template_manager import TemplateManager
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer
from forecasting_tools.data_models.questions import BinaryQuestion


class PersonalityBenchmarks(unittest.TestCase):
    """Performance benchmarks for personality configurations."""

    @classmethod
    def setUpClass(cls):
        """Set up benchmark fixtures for the entire test suite."""
        # Create temporary directory for benchmark personalities
        cls.benchmark_dir = tempfile.mkdtemp()
        
        # Generate benchmark personalities
        cls._generate_benchmark_personalities()
        
        # Add benchmark directory to search paths
        os.environ["PERSONALITY_DIRS"] = cls.benchmark_dir
        
        # Initialize components
        cls.manager = PersonalityManager()
        cls.cache = PersonalityCache()
        cls.template_manager = TemplateManager()
        cls.prompt_optimizer = PromptOptimizer()
        
        # Generate benchmark templates
        cls._generate_benchmark_templates()
    
    @classmethod
    def tearDownClass(cls):
        """Tear down benchmark fixtures."""
        # Clean up
        import shutil
        shutil.rmtree(cls.benchmark_dir)
        
        # Remove environment variable
        if "PERSONALITY_DIRS" in os.environ:
            del os.environ["PERSONALITY_DIRS"]
        
        # Clear caches
        cls.cache.invalidate_all()
        cls.template_manager.invalidate_all_templates()
        cls.prompt_optimizer.clear_cache()
    
    @classmethod
    def _generate_benchmark_personalities(cls):
        """Generate personalities for benchmarking."""
        # Thinking style variations
        thinking_styles = [
            ThinkingStyle.ANALYTICAL.value,
            ThinkingStyle.CREATIVE.value,
            ThinkingStyle.BALANCED.value,
            ThinkingStyle.BAYESIAN.value
        ]
        
        # Uncertainty approach variations
        uncertainty_approaches = [
            UncertaintyApproach.CAUTIOUS.value,
            UncertaintyApproach.BALANCED.value,
            UncertaintyApproach.BOLD.value
        ]
        
        # Reasoning depth variations
        reasoning_depths = [
            ReasoningDepth.SHALLOW.value,
            ReasoningDepth.MODERATE.value,
            ReasoningDepth.DEEP.value,
            ReasoningDepth.EXHAUSTIVE.value
        ]
        
        # Generate basic personalities
        for style in thinking_styles:
            personality_config = {
                "name": f"benchmark_{style}",
                "description": f"Benchmark personality with {style} thinking style",
                "thinking_style": style,
                "uncertainty_approach": UncertaintyApproach.BALANCED.value,
                "reasoning_depth": ReasoningDepth.MODERATE.value,
                "temperature": 0.7
            }
            
            with open(os.path.join(cls.benchmark_dir, f"benchmark_{style}.json"), "w") as f:
                json.dump(personality_config, f)
        
        # Generate uncertainty approach variations
        for approach in uncertainty_approaches:
            personality_config = {
                "name": f"benchmark_{approach}_uncertainty",
                "description": f"Benchmark personality with {approach} uncertainty approach",
                "thinking_style": ThinkingStyle.BALANCED.value,
                "uncertainty_approach": approach,
                "reasoning_depth": ReasoningDepth.MODERATE.value,
                "temperature": 0.7
            }
            
            with open(os.path.join(cls.benchmark_dir, f"benchmark_{approach}_uncertainty.json"), "w") as f:
                json.dump(personality_config, f)
        
        # Generate reasoning depth variations
        for depth in reasoning_depths:
            personality_config = {
                "name": f"benchmark_{depth}_depth",
                "description": f"Benchmark personality with {depth} reasoning depth",
                "thinking_style": ThinkingStyle.BALANCED.value,
                "uncertainty_approach": UncertaintyApproach.BALANCED.value,
                "reasoning_depth": depth,
                "temperature": 0.7
            }
            
            with open(os.path.join(cls.benchmark_dir, f"benchmark_{depth}_depth.json"), "w") as f:
                json.dump(personality_config, f)
        
        # Generate personalities with custom traits and varying complexity
        for i in range(5):
            num_traits = i + 1
            traits = {}
            
            for j in range(num_traits):
                trait_name = f"custom_trait_{j}"
                traits[trait_name] = {
                    "name": trait_name,
                    "description": f"Custom trait {j}",
                    "value": random.random()
                }
            
            personality_config = {
                "name": f"benchmark_custom_{i}",
                "description": f"Benchmark personality with {num_traits} custom traits",
                "thinking_style": random.choice(thinking_styles),
                "uncertainty_approach": random.choice(uncertainty_approaches),
                "reasoning_depth": random.choice(reasoning_depths),
                "temperature": 0.5 + (random.random() * 0.5),
                "traits": traits,
                "template_variables": {
                    f"var_{j}": f"value_{j}" for j in range(i)
                }
            }
            
            with open(os.path.join(cls.benchmark_dir, f"benchmark_custom_{i}.json"), "w") as f:
                json.dump(personality_config, f)
    
    @classmethod
    def _generate_benchmark_templates(cls):
        """Generate templates for benchmarking."""
        # Create template directory
        template_dir = os.path.join(cls.benchmark_dir, "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # Add template directory
        cls.template_manager.add_template_directory(template_dir)
        
        # Create simple template
        simple_template = {
            "content": "This is a simple template with {{variable}}.",
            "variables": {
                "variable": "default value"
            }
        }
        
        with open(os.path.join(template_dir, "simple_template.json"), "w") as f:
            json.dump(simple_template, f)
        
        # Create complex template with conditionals
        complex_template = {
            "content": """This is a complex template with multiple variables:
            
            Variable 1: {{var1}}
            Variable 2: {{var2}}
            
            <!-- IF condition == true -->
            This section is conditional and will only appear if condition is true.
            Additional content: {{conditional_var}}
            <!-- ENDIF -->
            
            Final section with {{var3}}.
            """,
            "variables": {
                "var1": "default 1",
                "var2": "default 2",
                "var3": "default 3",
                "condition": "false",
                "conditional_var": "conditional content"
            }
        }
        
        with open(os.path.join(template_dir, "complex_template.json"), "w") as f:
            json.dump(complex_template, f)
        
        # Create a large template
        large_content = "This is a large template.\n\n"
        for i in range(100):
            large_content += f"Line {i}: Some content with {{{{var_{i}}}}}\n"
        
        large_template = {
            "content": large_content,
            "variables": {
                f"var_{i}": f"value_{i}" for i in range(100)
            }
        }
        
        with open(os.path.join(template_dir, "large_template.json"), "w") as f:
            json.dump(large_template, f)

    def test_benchmark_loading_time(self):
        """Benchmark personality loading time with and without caching."""
        # Get all benchmark personalities
        personalities = [p for p in self.manager.list_available_personalities() if p.startswith("benchmark_")]
        print(f"\nBenchmarking loading time for {len(personalities)} personalities")
        
        # First, measure time without cache (clear it first)
        self.cache.invalidate_all()
        
        # Measure time to load all personalities without cache
        uncached_times = []
        for name in personalities:
            start_time = time.time()
            self.manager.load_personality(name)
            end_time = time.time()
            uncached_times.append(end_time - start_time)
        
        avg_uncached_time = sum(uncached_times) / len(uncached_times)
        print(f"Average loading time without cache: {avg_uncached_time:.6f} seconds")
        
        # Now measure time with cache
        cached_times = []
        for name in personalities:
            start_time = time.time()
            self.manager.load_personality(name)
            end_time = time.time()
            cached_times.append(end_time - start_time)
        
        avg_cached_time = sum(cached_times) / len(cached_times)
        print(f"Average loading time with cache: {avg_cached_time:.6f} seconds")
        
        # Calculate improvement
        if avg_uncached_time > 0:
            improvement = ((avg_uncached_time - avg_cached_time) / avg_uncached_time) * 100
            print(f"Cache improvement: {improvement:.2f}%")
        
        # Verify cache is faster
        self.assertLess(avg_cached_time, avg_uncached_time)
        
        # Get cache stats
        stats = self.cache.get_stats()
        print(f"Cache stats: {stats}")

    def test_benchmark_template_loading(self):
        """Benchmark template loading with and without lazy loading."""
        # Discover templates
        templates = self.template_manager.discover_templates()
        print(f"\nBenchmarking template loading for {len(templates)} templates")
        
        # Clear template cache
        self.template_manager.invalidate_all_templates()
        
        # Measure initial loading time for all templates
        initial_times = []
        for name in templates:
            start_time = time.time()
            self.template_manager.get_template(name)
            end_time = time.time()
            initial_times.append(end_time - start_time)
        
        avg_initial_time = sum(initial_times) / len(initial_times)
        print(f"Average initial template loading time: {avg_initial_time:.6f} seconds")
        
        # Measure subsequent loading time (should use cache)
        subsequent_times = []
        for name in templates:
            start_time = time.time()
            self.template_manager.get_template(name)
            end_time = time.time()
            subsequent_times.append(end_time - start_time)
        
        avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
        print(f"Average subsequent template loading time: {avg_subsequent_time:.6f} seconds")
        
        # Calculate improvement
        if avg_initial_time > 0:
            improvement = ((avg_initial_time - avg_subsequent_time) / avg_initial_time) * 100
            print(f"Lazy loading improvement: {improvement:.2f}%")
        
        # Verify lazy loading is faster
        self.assertLess(avg_subsequent_time, avg_initial_time)

    def test_benchmark_prompt_generation(self):
        """Benchmark prompt generation with different personality complexities."""
        # Get personality names from each category
        thinking_styles = [f"benchmark_{style}" for style in ['analytical', 'creative', 'balanced', 'bayesian']]
        custom_personalities = [f"benchmark_custom_{i}" for i in range(5)]
        
        # Test variables
        variables = {"variable": "test value", "condition": "true", "conditional_var": "test conditional"}
        
        print("\nBenchmarking prompt generation with different personalities")
        
        # Benchmark simple template with different thinking styles
        print("\nSimple template with different thinking styles:")
        for name in thinking_styles:
            times = []
            for _ in range(5):  # Run multiple times for better average
                start_time = time.time()
                self.prompt_optimizer.generate_prompt("simple_template", name, variables, compress=False)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            print(f"- {name}: {avg_time:.6f} seconds")
        
        # Benchmark complex template with different personality complexities
        print("\nComplex template with different personality complexities:")
        for name in custom_personalities:
            times = []
            for _ in range(5):  # Run multiple times for better average
                start_time = time.time()
                self.prompt_optimizer.generate_prompt("complex_template", name, variables, compress=False)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            print(f"- {name}: {avg_time:.6f} seconds")
        
        # Benchmark with and without compression
        print("\nPrompt generation with and without compression (large template):")
        for compress in [False, True]:
            times = []
            for _ in range(5):
                start_time = time.time()
                self.prompt_optimizer.generate_prompt("large_template", "benchmark_balanced", variables, compress=compress)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            print(f"- Compress={compress}: {avg_time:.6f} seconds")

    def test_benchmark_prompt_cache(self):
        """Benchmark prompt caching in the optimizer."""
        # Clear cache first
        self.prompt_optimizer.clear_cache()
        
        # Test variables
        variables = {"variable": "test value", "condition": "true", "conditional_var": "test conditional"}
        
        print("\nBenchmarking prompt caching")
        
        # Measure time for initial generation (uncached)
        uncached_times = []
        for _ in range(10):
            start_time = time.time()
            self.prompt_optimizer.generate_prompt("complex_template", "benchmark_analytical", variables, compress=False)
            end_time = time.time()
            uncached_times.append(end_time - start_time)
        
        avg_uncached_time = sum(uncached_times) / len(uncached_times)
        print(f"Average generation time without cache: {avg_uncached_time:.6f} seconds")
        
        # Measure time for cached generation
        cached_times = []
        for _ in range(10):
            start_time = time.time()
            self.prompt_optimizer.generate_prompt("complex_template", "benchmark_analytical", variables, compress=False)
            end_time = time.time()
            cached_times.append(end_time - start_time)
        
        avg_cached_time = sum(cached_times) / len(cached_times)
        print(f"Average generation time with cache: {avg_cached_time:.6f} seconds")
        
        # Calculate improvement
        if avg_uncached_time > 0:
            improvement = ((avg_uncached_time - avg_cached_time) / avg_uncached_time) * 100
            print(f"Cache improvement: {improvement:.2f}%")
        
        # Verify cache is faster
        self.assertLess(avg_cached_time, avg_uncached_time)
        
        # Get cache stats
        stats = self.prompt_optimizer.get_cache_stats()
        print(f"Prompt cache stats: {stats}")

    def test_benchmark_prompt_pipeline(self):
        """Benchmark the full prompt optimization pipeline."""
        # Set of templates and personalities to test
        templates = ["simple_template", "complex_template", "large_template"]
        personalities = ["benchmark_analytical", "benchmark_creative", "benchmark_cautious_uncertainty"]
        
        # Test variables
        variables = {
            "variable": "test value", 
            "condition": "true", 
            "conditional_var": "test conditional",
            "var1": "test 1",
            "var2": "test 2",
            "var3": "test 3"
        }
        
        print("\nBenchmarking full prompt optimization pipeline")
        
        # Test with different combinations
        results = {}
        for template in templates:
            for personality in personalities:
                key = f"{personality} + {template}"
                times = []
                
                for _ in range(3):  # Run multiple times for better average
                    start_time = time.time()
                    _, metadata = self.prompt_optimizer.optimize_prompt_pipeline(
                        personality_name=personality,
                        template_name=template,
                        variables=variables,
                        context_size=None  # No compression
                    )
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                results[key] = {
                    "avg_time": avg_time,
                    "tokens": metadata["estimated_tokens"]
                }
        
        # Print results
        print("\nPrompt pipeline results:")
        for key, data in results.items():
            print(f"- {key}: {data['avg_time']:.6f} seconds, ~{data['tokens']} tokens")
        
        # Test with and without context size limitation (compression)
        print("\nTesting with and without context size limitation:")
        for context_size in [None, 1000, 500]:
            times = []
            for _ in range(3):
                start_time = time.time()
                _, metadata = self.prompt_optimizer.optimize_prompt_pipeline(
                    personality_name="benchmark_deep_depth",
                    template_name="large_template",
                    variables=variables,
                    context_size=context_size
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            print(f"- Context size {context_size}: {avg_time:.6f} seconds, ~{metadata['estimated_tokens']} tokens")


if __name__ == "__main__":
    unittest.main() 