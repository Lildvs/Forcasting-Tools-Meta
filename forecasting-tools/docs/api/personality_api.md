# Personality Management API Reference

This document provides a comprehensive reference for the Personality Management API, covering all classes, methods, and configuration options available to users.

## Table of Contents

- [PersonalityManager](#personalitymanager)
- [PersonalityConfig](#personalityconfig)
- [TemplateManager](#templatemanager)
- [PromptOptimizer](#promptoptimizer)
- [PersonalityCache](#personalitycache)
- [PersonalityValidator](#personalityvalidator)
- [PersonalityDebugger](#personalitydebugger)
- [PersonalityTelemetryTracker](#personalitytelemetrytracker)
- [PersonalityFeatureFlags](#personalityfeatureflags)
- [Enumerations](#enumerations)
- [Configuration Schema](#configuration-schema)
- [Template Format](#template-format)
- [Integration API](#integration-api)

## PersonalityManager

The central manager for loading, accessing, and managing personalities.

### Class: `PersonalityManager`

```python
from forecasting_tools.personality_management import PersonalityManager

manager = PersonalityManager()
```

#### Methods

##### `list_available_personalities()`

Returns a list of available personality names.

```python
personalities = manager.list_available_personalities()
# ['analytical', 'creative', 'balanced', ...]
```

##### `load_personality(personality_name: str) -> PersonalityConfig`

Loads a personality by name.

```python
personality = manager.load_personality("analytical")
```

##### `load_personality_from_file(file_path: str) -> PersonalityConfig`

Loads a personality from a specific file path.

```python
personality = manager.load_personality_from_file("/path/to/custom_personality.json")
```

##### `create_personality(config_dict: Dict[str, Any]) -> PersonalityConfig`

Creates a personality from a configuration dictionary.

```python
config = {
    "name": "custom",
    "thinking_style": "analytical",
    "uncertainty_approach": "cautious",
    "reasoning_depth": "deep"
}
personality = manager.create_personality(config)
```

##### `register_personality(personality: PersonalityConfig) -> bool`

Registers a personality for later use.

```python
manager.register_personality(personality)
```

##### `add_personality_directory(directory: str) -> None`

Adds a directory to search for personality configurations.

```python
manager.add_personality_directory("./my_personalities")
```

## PersonalityConfig

Represents a personality configuration with thinking style, uncertainty approach, and other traits.

### Class: `PersonalityConfig`

```python
from forecasting_tools.personality_management.config import PersonalityConfig
```

#### Attributes

- `name: str`: Name of the personality
- `description: Optional[str]`: Description of the personality
- `thinking_style: ThinkingStyle`: Thinking style (analytical, creative, etc.)
- `uncertainty_approach: UncertaintyApproach`: Uncertainty approach (cautious, bold, etc.)
- `reasoning_depth: ReasoningDepth`: Reasoning depth (shallow, deep, etc.)
- `temperature: float`: Temperature setting (default: 0.7)
- `traits: Dict[str, PersonalityTrait]`: Custom traits
- `template_variables: Dict[str, Any]`: Variables for templates

#### Methods

##### `from_dict(config_dict: Dict[str, Any]) -> PersonalityConfig`

Creates a personality from a dictionary.

```python
config_dict = {
    "name": "custom",
    "thinking_style": "analytical",
    "uncertainty_approach": "cautious",
    "reasoning_depth": "deep"
}
personality = PersonalityConfig.from_dict(config_dict)
```

##### `to_dict() -> Dict[str, Any]`

Converts the personality to a dictionary.

```python
config_dict = personality.to_dict()
```

##### `get_trait(trait_name: str) -> Optional[PersonalityTrait]`

Gets a specific trait by name.

```python
creativity_trait = personality.get_trait("creativity")
```

### Class: `PersonalityTrait`

Represents a custom trait for a personality.

```python
from forecasting_tools.personality_management.config import PersonalityTrait
```

#### Attributes

- `name: str`: Name of the trait
- `description: str`: Description of the trait
- `value: Any`: Value of the trait (numeric, string, or boolean)

#### Methods

##### `from_dict(trait_dict: Dict[str, Any]) -> PersonalityTrait`

Creates a trait from a dictionary.

```python
trait_dict = {
    "name": "creativity",
    "description": "Level of creative thinking",
    "value": 0.8
}
trait = PersonalityTrait.from_dict(trait_dict)
```

##### `to_dict() -> Dict[str, Any]`

Converts the trait to a dictionary.

```python
trait_dict = trait.to_dict()
```

## TemplateManager

Manages templates for personalities with lazy loading capabilities.

### Class: `TemplateManager`

```python
from forecasting_tools.personality_management.template_manager import TemplateManager

template_manager = TemplateManager()
```

#### Methods

##### `add_template_directory(directory: str) -> None`

Adds a directory to search for templates.

```python
template_manager.add_template_directory("./my_templates")
```

##### `discover_templates() -> List[str]`

Discovers available templates without loading them.

```python
templates = template_manager.discover_templates()
# ['forecast_template', 'analysis_template', ...]
```

##### `get_template(template_name: str, force_reload: bool = False) -> Optional[Dict[str, Any]]`

Gets a template, loading it if necessary.

```python
template = template_manager.get_template("forecast_template")
```

##### `get_template_field(template_name: str, field_name: str) -> Any`

Gets a specific field from a template.

```python
content = template_manager.get_template_field("forecast_template", "content")
```

##### `combine_templates(base_template: str, extension_templates: List[str]) -> Dict[str, Any]`

Combines multiple templates into one.

```python
combined = template_manager.combine_templates(
    "base_template", 
    ["extension1", "extension2"]
)
```

##### `render_template(template_name: str, variables: Dict[str, Any]) -> Optional[str]`

Renders a template with variables.

```python
rendered = template_manager.render_template(
    "forecast_template",
    {"question": "Will GDP grow?", "thinking_style": "analytical"}
)
```

##### `invalidate_template(template_name: str) -> None`

Invalidates a template in the cache.

```python
template_manager.invalidate_template("forecast_template")
```

##### `invalidate_all_templates() -> None`

Invalidates all templates in the cache.

```python
template_manager.invalidate_all_templates()
```

## PromptOptimizer

Optimizer for prompt generation pipelines.

### Class: `PromptOptimizer`

```python
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer()
```

#### Methods

##### `generate_prompt(prompt_template: str, personality_name: Optional[str] = None, variables: Optional[Dict[str, Any]] = None, compress: bool = False, use_cache: bool = True) -> str`

Generates an optimized prompt using a template and personality.

```python
prompt = optimizer.generate_prompt(
    "forecast_template",
    "analytical",
    {"question": "Will GDP grow?"},
    compress=True
)
```

##### `optimize_prompt_pipeline(personality_name: str, template_name: str, variables: Dict[str, Any], context_size: Optional[int] = None) -> Tuple[str, Dict[str, Any]]`

Optimizes the full prompt generation pipeline.

```python
prompt, metadata = optimizer.optimize_prompt_pipeline(
    "analytical",
    "forecast_template",
    {"question": "Will GDP grow?"},
    context_size=2000
)
```

##### `set_max_cache_size(size: int) -> None`

Sets the maximum prompt cache size.

```python
optimizer.set_max_cache_size(100)
```

##### `clear_cache() -> None`

Clears the prompt cache.

```python
optimizer.clear_cache()
```

##### `get_cache_stats() -> Dict[str, Any]`

Gets cache statistics.

```python
stats = optimizer.get_cache_stats()
```

## PersonalityCache

Cache for personality configurations to improve performance.

### Class: `PersonalityCache`

```python
from forecasting_tools.personality_management.cache import PersonalityCache

cache = PersonalityCache()
```

#### Methods

##### `set_ttl(ttl_seconds: int) -> None`

Sets the time-to-live for cache entries.

```python
cache.set_ttl(300)  # 5 minutes
```

##### `get(personality_name: str) -> Optional[PersonalityConfig]`

Gets a personality configuration from the cache.

```python
personality = cache.get("analytical")
```

##### `put(personality_name: str, config: PersonalityConfig, file_path: Optional[str] = None) -> None`

Stores a personality configuration in the cache.

```python
cache.put("analytical", personality, "/path/to/analytical.json")
```

##### `invalidate(personality_name: str) -> None`

Invalidates a specific personality in the cache.

```python
cache.invalidate("analytical")
```

##### `invalidate_all() -> None`

Invalidates the entire cache.

```python
cache.invalidate_all()
```

##### `get_stats() -> Dict[str, Any]`

Gets cache statistics.

```python
stats = cache.get_stats()
```

### Decorator: `cached_personality`

Decorator for caching personality loading functions.

```python
from forecasting_tools.personality_management.cache import cached_personality

@cached_personality
def load_custom_personality(name: str) -> PersonalityConfig:
    # Loading logic here
    return personality
```

## PersonalityValidator

Validator for personality configurations.

### Class: `PersonalityValidator`

```python
from forecasting_tools.personality_management.validators import PersonalityValidator

validator = PersonalityValidator()
```

#### Methods

##### `validate_personality(personality: Union[PersonalityConfig, Dict[str, Any]]) -> Tuple[bool, List[str]]`

Validates a personality configuration.

```python
is_valid, errors = validator.validate_personality(personality)
```

##### `check_file_integrity(file_path: str) -> Tuple[bool, List[str]]`

Checks if a personality configuration file is valid.

```python
is_valid, errors = validator.check_file_integrity("/path/to/personality.json")
```

##### `validate_template_compatibility(personality: PersonalityConfig, template_name: str) -> Tuple[bool, List[str]]`

Checks if a personality is compatible with a template.

```python
is_compatible, issues = validator.validate_template_compatibility(
    personality, 
    "forecast_template"
)
```

##### `check_consistency(personality: PersonalityConfig) -> Tuple[bool, List[str]]`

Checks for internal consistency of a personality.

```python
is_consistent, issues = validator.check_consistency(personality)
```

##### `validate_personality_directory(directory: str) -> Dict[str, Dict[str, Any]]`

Validates all personality configurations in a directory.

```python
results = validator.validate_personality_directory("./personalities")
```

##### `generate_validation_report(validation_results: Dict[str, Dict[str, Any]], include_passes: bool = False) -> str`

Generates a readable validation report from validation results.

```python
report = validator.generate_validation_report(results, include_passes=True)
```

### Function: `validate_personality_file(file_path: str) -> bool`

Validates a personality configuration file.

```python
is_valid = validate_personality_file("/path/to/personality.json")
```

## PersonalityDebugger

Debugger for personality configurations and template integrations.

### Class: `PersonalityDebugger`

```python
from forecasting_tools.personality_management.debugger import PersonalityDebugger

debugger = PersonalityDebugger()
```

#### Methods

##### `set_log_level(level: int) -> None`

Sets the log level for the debugger.

```python
import logging
debugger.set_log_level(logging.DEBUG)
```

##### `debug(message: str, data: Optional[Dict[str, Any]] = None) -> None`

Logs a debug message.

```python
debugger.debug("Testing personality", {"name": "analytical"})
```

##### `get_logs(level: Optional[int] = None) -> List[Dict[str, Any]]`

Gets filtered debug logs.

```python
logs = debugger.get_logs(logging.WARNING)
```

##### `clear_logs() -> None`

Clears all debug logs.

```python
debugger.clear_logs()
```

##### `export_logs(file_path: str) -> bool`

Exports debug logs to a JSON file.

```python
debugger.export_logs("debug_logs.json")
```

##### `diagnose_personality(personality: Union[PersonalityConfig, str, Dict[str, Any]]) -> Dict[str, Any]`

Diagnoses issues with a personality configuration.

```python
diagnosis = debugger.diagnose_personality("analytical")
```

##### `test_template_rendering(personality_name: str, template_name: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Tests template rendering with a personality.

```python
results = debugger.test_template_rendering(
    "analytical",
    "forecast_template",
    {"question": "Will GDP grow?"}
)
```

##### `analyze_prompt_pipeline(personality_name: str, template_name: str, variables: Optional[Dict[str, Any]] = None, context_size: Optional[int] = None) -> Dict[str, Any]`

Analyzes the full prompt generation pipeline.

```python
results = debugger.analyze_prompt_pipeline(
    "analytical",
    "forecast_template",
    {"question": "Will GDP grow?"}
)
```

##### `simulate_bot_integration(personality_name: str, template_name: str, variables: Optional[Dict[str, Any]] = None, mock_response: Optional[str] = None) -> Dict[str, Any]`

Simulates integration with a forecasting bot.

```python
results = debugger.simulate_bot_integration(
    "analytical",
    "forecast_template",
    {"question": "Will GDP grow?"}
)
```

##### `analyze_cache_performance(clear_cache: bool = False) -> Dict[str, Any]`

Analyzes the performance of the cache.

```python
stats = debugger.analyze_cache_performance()
```

##### `generate_debug_report(output_file: Optional[str] = None) -> Dict[str, Any]`

Generates a comprehensive debug report.

```python
report = debugger.generate_debug_report("debug_report.json")
```

### Function: `get_debugger() -> PersonalityDebugger`

Gets the singleton debugger instance.

```python
from forecasting_tools.personality_management.debugger import get_debugger

debugger = get_debugger()
```

## PersonalityTelemetryTracker

Tracker for personality usage and performance metrics.

### Class: `PersonalityTelemetryTracker`

```python
from forecasting_tools.personality_management.telemetry import PersonalityTelemetryTracker

tracker = PersonalityTelemetryTracker()
```

#### Methods

##### `record_usage(personality_name: str, context: Optional[str] = None, duration_ms: Optional[int] = None, template_name: Optional[str] = None, token_count: Optional[int] = None, session_id: Optional[str] = None) -> None`

Records usage of a personality.

```python
tracker.record_usage(
    "analytical",
    context="forecast",
    duration_ms=150,
    template_name="forecast_template",
    token_count=1024
)
```

##### `record_performance(personality_name: str, metric_name: str, metric_value: float, domain: Optional[str] = None, question_id: Optional[str] = None) -> None`

Records performance metric for a personality.

```python
tracker.record_performance(
    "analytical",
    "calibration_score",
    0.85,
    domain="finance"
)
```

##### `record_anomaly(personality_name: str, anomaly_type: str, description: str, severity: str, metric_name: Optional[str] = None, metric_value: Optional[float] = None, expected_range: Optional[str] = None) -> None`

Records an anomaly in personality behavior.

```python
tracker.record_anomaly(
    "analytical",
    "high_value",
    "Unusually high confidence score",
    "medium",
    metric_name="confidence",
    metric_value=0.95,
    expected_range="0.6-0.8"
)
```

##### `get_usage_statistics(days: Optional[int] = None, personality_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]`

Gets usage statistics for personalities.

```python
stats = tracker.get_usage_statistics(days=30)
```

##### `get_performance_metrics(personality_name: Optional[str] = None, metric_name: Optional[str] = None, domain: Optional[str] = None, days: Optional[int] = None) -> Dict[str, Dict[str, float]]`

Gets aggregated performance metrics.

```python
metrics = tracker.get_performance_metrics(days=30)
```

##### `get_anomalies(personality_name: Optional[str] = None, severity: Optional[str] = None, days: Optional[int] = None) -> List[Dict[str, Any]]`

Gets recorded anomalies.

```python
anomalies = tracker.get_anomalies(severity="high")
```

##### `generate_performance_report(days: int = 30, personality_name: Optional[str] = None) -> Dict[str, Any]`

Generates a comprehensive performance report.

```python
report = tracker.generate_performance_report(days=30)
```

##### `export_data(export_path: str) -> bool`

Exports telemetry data to a JSON file.

```python
tracker.export_data("telemetry_export.json")
```

### Functions

##### `get_telemetry_tracker() -> PersonalityTelemetryTracker`

Gets the singleton telemetry tracker instance.

```python
from forecasting_tools.personality_management.telemetry import get_telemetry_tracker

tracker = get_telemetry_tracker()
```

##### `record_personality_usage(personality_name: str, ...) -> None`

Records personality usage.

```python
from forecasting_tools.personality_management.telemetry import record_personality_usage

record_personality_usage("analytical", context="forecast")
```

##### `record_personality_performance(personality_name: str, metric_name: str, metric_value: float, ...) -> None`

Records personality performance metric.

```python
from forecasting_tools.personality_management.telemetry import record_personality_performance

record_personality_performance("analytical", "calibration", 0.85)
```

##### `get_personality_report(days: int = 30, personality_name: Optional[str] = None) -> Dict[str, Any]`

Generates a personality performance report.

```python
from forecasting_tools.personality_management.telemetry import get_personality_report

report = get_personality_report(days=30)
```

## PersonalityFeatureFlags

Feature flag manager for the personality system.

### Class: `PersonalityFeatureFlags`

```python
from forecasting_tools.personality_management.feature_flags import PersonalityFeatureFlags

flags = PersonalityFeatureFlags()
```

#### Methods

##### `is_enabled(flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool`

Checks if a feature flag is enabled.

```python
if flags.is_enabled("personality_system_enabled"):
    # System is enabled
```

##### `get_all_flags() -> Dict[str, Any]`

Gets all feature flags.

```python
all_flags = flags.get_all_flags()
```

##### `update_flag(flag_name: str, value: Any) -> bool`

Updates a feature flag value (runtime only).

```python
flags.update_flag("verbose_logging", True)
```

##### `save_configuration(file_path: Optional[str] = None) -> bool`

Saves current feature flag configuration to a file.

```python
flags.save_configuration()
```

### Function: `get_feature_flags() -> PersonalityFeatureFlags`

Gets the singleton feature flags instance.

```python
from forecasting_tools.personality_management.feature_flags import get_feature_flags

flags = get_feature_flags()
```

## Enumerations

### ThinkingStyle

```python
from forecasting_tools.personality_management.config import ThinkingStyle

style = ThinkingStyle.ANALYTICAL
# Available values:
# - ThinkingStyle.ANALYTICAL
# - ThinkingStyle.CREATIVE
# - ThinkingStyle.BALANCED
# - ThinkingStyle.BAYESIAN
```

### UncertaintyApproach

```python
from forecasting_tools.personality_management.config import UncertaintyApproach

approach = UncertaintyApproach.CAUTIOUS
# Available values:
# - UncertaintyApproach.CAUTIOUS
# - UncertaintyApproach.BALANCED
# - UncertaintyApproach.BOLD
```

### ReasoningDepth

```python
from forecasting_tools.personality_management.config import ReasoningDepth

depth = ReasoningDepth.DEEP
# Available values:
# - ReasoningDepth.SHALLOW
# - ReasoningDepth.MODERATE
# - ReasoningDepth.DEEP
# - ReasoningDepth.EXHAUSTIVE
```

## Configuration Schema

### Personality Configuration Schema

```json
{
  "name": "String (required)",
  "description": "String (optional)",
  "thinking_style": "String: 'analytical', 'creative', 'balanced', or 'bayesian' (required)",
  "uncertainty_approach": "String: 'cautious', 'balanced', or 'bold' (required)",
  "reasoning_depth": "String: 'shallow', 'moderate', 'deep', or 'exhaustive' (required)",
  "temperature": "Number (optional, default: 0.7)",
  "traits": {
    "trait_name": {
      "name": "String (should match the key)",
      "description": "String (optional)",
      "value": "Number, String, or Boolean"
    }
  },
  "template_variables": {
    "variable_name": "value"
  }
}
```

## Template Format

### Template Schema

```json
{
  "content": "String with {{variable}} placeholders (required)",
  "variables": {
    "variable_name": "default_value"
  }
}
```

### Conditional Sections

Templates support conditional sections with this syntax:

```
<!-- IF variable_name == value -->
Content to include if condition is true
<!-- ENDIF -->
```

## Integration API

### ForecastingBot Integration

```python
from forecasting_tools import ForecastingBot

# Create bot with personality
bot = ForecastingBot(personality_name="analytical")

# Use bot normally
forecast = bot.forecast_binary(question)
```

### Custom LLM Integration

```python
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer

# Load personality
manager = PersonalityManager()
personality = manager.load_personality("analytical")

# Get temperature setting
temperature = personality.temperature

# Generate prompt
optimizer = PromptOptimizer()
prompt, metadata = optimizer.optimize_prompt_pipeline(
    personality_name="analytical",
    template_name="my_custom_template",
    variables={"question": "My forecast question"}
)

# Use with your LLM
# llm.generate(prompt, temperature=temperature)
``` 