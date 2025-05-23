# Testing Custom Personalities

This document outlines the procedures and best practices for testing custom personalities in the forecasting tools system. A comprehensive testing approach ensures that new personalities perform as expected and integrate well with the forecasting pipeline.

## 1. Testing Framework

The forecasting tools package includes a robust testing framework for personalities, located in the `forecasting-tools/code_tests/personality/` directory. This framework consists of:

- **Unit tests**: Verify individual personality components
- **Integration tests**: Check personality effects on forecasts
- **Performance benchmarks**: Measure performance characteristics
- **Validation utilities**: Ensure personality configuration correctness

## 2. Testing Procedure for Custom Personalities

Follow this step-by-step procedure when testing a new custom personality:

### 2.1. Configuration Validation

Start by validating the personality configuration using the validation utilities:

```python
from forecasting_tools.personality_management.validators import PersonalityValidator

validator = PersonalityValidator()

# Validate a personality configuration file
is_valid, errors = validator.check_file_integrity("path/to/your_personality.json")

if not is_valid:
    for error in errors:
        print(f"Error: {error}")
else:
    print("Configuration is valid!")
    
# Check consistency of traits and parameters
from forecasting_tools.personality_management import PersonalityManager
manager = PersonalityManager()
personality = manager.load_personality("your_personality_name")

is_consistent, issues = validator.check_consistency(personality)
if not is_consistent:
    for issue in issues:
        print(f"Consistency issue: {issue}")
```

### 2.2. Template Compatibility Testing

Verify that your personality is compatible with the templates it will use:

```python
# Check compatibility with a specific template
is_compatible, issues = validator.validate_template_compatibility(
    personality, 
    "forecast_template"
)

if not is_compatible:
    for issue in issues:
        print(f"Compatibility issue: {issue}")
else:
    print("Personality is compatible with the template!")
    
# Test rendering with the template
from forecasting_tools.personality_management.debugger import PersonalityDebugger

debugger = PersonalityDebugger()
results = debugger.test_template_rendering(
    personality_name="your_personality_name",
    template_name="forecast_template",
    variables={"question": "Will GDP grow next quarter?"}
)

if results["success"]:
    print("Template rendered successfully!")
    print(f"Rendered content length: {len(results['rendered_content'])}")
else:
    print(f"Error: {results['error']}")
```

### 2.3. Basic Unit Testing

Write unit tests for your custom personality:

```python
import unittest
from forecasting_tools.personality_management import PersonalityManager

class TestCustomPersonality(unittest.TestCase):
    
    def setUp(self):
        self.manager = PersonalityManager()
        self.personality = self.manager.load_personality("your_personality_name")
    
    def test_personality_traits(self):
        """Test basic personality traits."""
        self.assertEqual(self.personality.thinking_style, "analytical")
        self.assertEqual(self.personality.uncertainty_approach, "cautious")
        self.assertEqual(self.personality.reasoning_depth, "deep")
        
        # Check custom traits
        self.assertIn("domain_expertise", self.personality.traits)
        self.assertEqual(self.personality.traits["domain_expertise"].value, 0.8)
    
    def test_template_variables(self):
        """Test template variables."""
        self.assertIn("custom_instructions", self.personality.template_variables)
        self.assertTrue(len(self.personality.template_variables["custom_instructions"]) > 0)

if __name__ == "__main__":
    unittest.main()
```

### 2.4. Integration Testing

Test how your personality affects forecast outputs:

```python
from forecasting_tools.code_tests.personality.test_integration import MockForecaster
from forecasting_tools.data_models.questions import BinaryQuestion

# Create a test question
question = BinaryQuestion(
    question_text="Will GDP grow next quarter?",
    background_info="Economic indicators are mixed, with inflation dropping but consumer spending decreasing.",
    resolution_criteria="GDP growth will be measured as reported by official government statistics.",
    fine_print="",
    page_url="",
    api_json={}
)

# Test forecasting with your personality
forecaster = MockForecaster("your_personality_name")
report = forecaster.forecast(question)

# Compare with a baseline personality
baseline_forecaster = MockForecaster("balanced")
baseline_report = baseline_forecaster.forecast(question)

# Output results for comparison
print(f"Custom personality probability: {report.binary_prob}")
print(f"Baseline personality probability: {baseline_report.binary_prob}")
print("\nCustom personality reasoning:")
print(report.reasoning)
```

### 2.5. Performance Benchmarking

Measure performance characteristics of your personality:

```python
from forecasting_tools.code_tests.personality.test_performance_benchmarks import PersonalityBenchmarks

benchmarks = PersonalityBenchmarks()

# Add your personality to the benchmark set
benchmarks._generate_benchmark_personalities()

# Run specific benchmarks
benchmarks.test_benchmark_prompt_generation()
benchmarks.test_benchmark_loading_time()
```

### 2.6. Prompt Pipeline Analysis

Analyze how your personality influences the prompt generation pipeline:

```python
from forecasting_tools.personality_management.debugger import PersonalityDebugger

debugger = PersonalityDebugger()
results = debugger.analyze_prompt_pipeline(
    personality_name="your_personality_name",
    template_name="forecast_template",
    variables={"question": "Will GDP grow next quarter?"}
)

print(f"Pipeline processing time: {results['timing']['pipeline_processing']}s")
print(f"Estimated token count: {results['metadata']['estimated_tokens']}")
```

### 2.7. Bot Integration Simulation

Simulate how your personality would interact with a forecasting bot:

```python
simulation_results = debugger.simulate_bot_integration(
    personality_name="your_personality_name",
    template_name="forecast_template",
    variables={"question": "Will GDP grow next quarter?"}
)

print("Generated prompt:")
print(simulation_results["prompt"])
print("\nSimulated response:")
print(simulation_results["response"])
```

## 3. Automated Test Suite

To run the complete automated test suite for your personality, create a test script:

```python
import unittest
import sys
import os

# Add the directory containing 'forecasting_tools' to the Python path
sys.path.insert(0, os.path.abspath("../../"))

from forecasting_tools.personality_management.validators import PersonalityValidator
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.debugger import PersonalityDebugger
from forecasting_tools.code_tests.personality.test_integration import MockForecaster
from forecasting_tools.data_models.questions import BinaryQuestion

class TestSuiteForCustomPersonality(unittest.TestCase):
    """Comprehensive test suite for custom personalities."""
    
    @classmethod
    def setUpClass(cls):
        cls.validator = PersonalityValidator()
        cls.manager = PersonalityManager()
        cls.debugger = PersonalityDebugger()
        cls.personality_name = "your_personality_name"
        
        # Load the personality
        cls.personality = cls.manager.load_personality(cls.personality_name)
        
        # Test question
        cls.test_question = BinaryQuestion(
            question_text="Will GDP grow next quarter?",
            background_info="Economic indicators are mixed.",
            resolution_criteria="Official statistics.",
            fine_print="",
            page_url="",
            api_json={}
        )
    
    def test_01_validation(self):
        """Test personality validation."""
        is_valid, errors = self.validator.validate_personality(self.personality)
        self.assertTrue(is_valid, f"Validation errors: {errors}")
    
    def test_02_consistency(self):
        """Test personality consistency."""
        is_consistent, issues = self.validator.check_consistency(self.personality)
        # Note: We don't assert this as True because some inconsistencies might be intentional
        if not is_consistent:
            print(f"Warning: Consistency issues found: {issues}")
    
    def test_03_template_compatibility(self):
        """Test template compatibility."""
        templates = ["forecast_template", "analysis_template"]
        for template_name in templates:
            is_compatible, issues = self.validator.validate_template_compatibility(
                self.personality, 
                template_name
            )
            self.assertTrue(is_compatible, f"Template compatibility issues with {template_name}: {issues}")
    
    def test_04_template_rendering(self):
        """Test template rendering."""
        for template_name in ["forecast_template", "analysis_template"]:
            results = self.debugger.test_template_rendering(
                personality_name=self.personality_name,
                template_name=template_name,
                variables={"question": "Will GDP grow next quarter?"}
            )
            self.assertTrue(results["success"], f"Template rendering failed: {results.get('error')}")
            self.assertIsNotNone(results["rendered_content"])
            self.assertTrue(len(results["rendered_content"]) > 0)
    
    def test_05_prompt_pipeline(self):
        """Test prompt pipeline."""
        results = self.debugger.analyze_prompt_pipeline(
            personality_name=self.personality_name,
            template_name="forecast_template",
            variables={"question": "Will GDP grow next quarter?"}
        )
        self.assertTrue(results["success"], f"Prompt pipeline analysis failed: {results.get('error')}")
        self.assertLess(results["timing"]["total"], 2.0, "Prompt generation taking too long")
    
    def test_06_forecast_generation(self):
        """Test forecast generation."""
        forecaster = MockForecaster(self.personality_name)
        report = forecaster.forecast(self.test_question)
        
        self.assertIsNotNone(report.binary_prob)
        self.assertTrue(0 <= report.binary_prob <= 1)
        self.assertIsNotNone(report.reasoning)
        self.assertTrue(len(report.reasoning) > 0)

if __name__ == "__main__":
    unittest.main()
```

Save this as a Python script and run it to execute the full test suite for your personality.

## 4. Testing Best Practices

Follow these best practices when testing custom personalities:

### 4.1. Regression Testing

When modifying an existing personality, always run a regression test to ensure the changes don't break existing functionality:

```python
# Before making changes, generate baseline forecasts
baseline_forecaster = MockForecaster("existing_personality")
baseline_results = [baseline_forecaster.forecast(q) for q in test_questions]

# After making changes, compare with the baseline
modified_forecaster = MockForecaster("modified_personality")
modified_results = [modified_forecaster.forecast(q) for q in test_questions]

# Compare results
for i, (baseline, modified) in enumerate(zip(baseline_results, modified_results)):
    print(f"Question {i+1}:")
    print(f"  Baseline: {baseline.binary_prob}")
    print(f"  Modified: {modified.binary_prob}")
    print(f"  Difference: {abs(baseline.binary_prob - modified.binary_prob)}")
```

### 4.2. Domain-Specific Testing

Test personalities with questions from their intended domain:

```python
# For a finance-focused personality
finance_questions = [
    BinaryQuestion(question_text="Will the S&P 500 close higher next month?", ...),
    BinaryQuestion(question_text="Will inflation exceed 3% next quarter?", ...),
    # Add more domain-specific questions
]

domain_forecaster = MockForecaster("finance_personality")
results = [domain_forecaster.forecast(q) for q in finance_questions]

# Evaluate domain expertise
for q, r in zip(finance_questions, results):
    print(f"Question: {q.question_text}")
    print(f"Probability: {r.binary_prob}")
    print(f"Reasoning includes financial terms: {'inflation' in r.reasoning or 'market' in r.reasoning}")
```

### 4.3. Comparative Testing

Compare your personality against established ones:

```python
personalities = ["analytical", "creative", "bayesian", "your_personality_name"]
forecasters = {name: MockForecaster(name) for name in personalities}

question = BinaryQuestion(question_text="Will remote work increase next year?", ...)

results = {name: forecaster.forecast(question) for name, forecaster in forecasters.items()}

# Compare probabilities
for name, report in results.items():
    print(f"{name}: {report.binary_prob}")

# Compare reasoning depth
for name, report in results.items():
    word_count = len(report.reasoning.split())
    print(f"{name} reasoning word count: {word_count}")
```

### 4.4. Edge Case Testing

Test your personality with edge case questions:

```python
edge_cases = [
    BinaryQuestion(question_text="Will an entirely unpredictable event occur?", ...),
    BinaryQuestion(question_text="Will a question with no background information be answerable?", background_info="", ...),
    BinaryQuestion(question_text="Will a question with extremely vague resolution criteria be forecasted well?", resolution_criteria="It depends.", ...)
]

edge_forecaster = MockForecaster("your_personality_name")
for question in edge_cases:
    report = edge_forecaster.forecast(question)
    print(f"Question: {question.question_text}")
    print(f"Probability: {report.binary_prob}")
    print(f"Uncertainty acknowledged: {'uncertain' in report.reasoning.lower()}")
```

## 5. Debugging Tools

When tests fail, use the debugging tools to diagnose issues:

```python
from forecasting_tools.personality_management.debugger import get_debugger

debugger = get_debugger()
debugger.set_log_level(logging.DEBUG)  # Set detailed logging

# Diagnose personality issues
diagnosis = debugger.diagnose_personality("your_personality_name")
if not diagnosis["success"]:
    print("Personality diagnosis failed!")
    for issue in diagnosis["issues"]:
        print(f"- {issue}")

# Generate a comprehensive debug report
report = debugger.generate_debug_report("debug_report.json")

# Check cache performance
cache_stats = debugger.analyze_cache_performance()
print(f"Cache hit rate: {cache_stats['personality_cache']['hit_rate']:.2f}")
```

## 6. Continuous Testing

For production environments, implement continuous testing to monitor personality performance over time:

```python
from forecasting_tools.personality_management.telemetry import get_telemetry_tracker

tracker = get_telemetry_tracker()

# Record personality usage and performance
tracker.record_usage(
    personality_name="your_personality_name",
    context="production",
    template_name="forecast_template"
)

# Record performance metrics
tracker.record_performance(
    personality_name="your_personality_name",
    metric_name="calibration_score",
    metric_value=0.85,
    domain="finance"
)

# Generate performance reports
report = tracker.generate_performance_report(days=30)
print(f"Total usage: {report['overall_statistics']['total_usage']}")
print(f"Anomalies detected: {len(report['anomalies'])}")
```

By following these testing procedures, you can ensure that your custom personalities perform reliably and effectively within the forecasting tools ecosystem. 