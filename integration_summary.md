# Personality-Enhanced Ensemble Forecasting System

This document provides an overview of the integration of personality management with the ensemble forecasting and competition frameworks.

## Key Components and Integration

### 1. Ensemble Forecasting

#### EnsembleBot (forecasting_tools/forecast_bots/ensemble_bot.py)
- Creates ensembles of forecasters with diverse personalities
- Dynamically manages bot selection and weighting based on personality traits
- Aggregates forecasts with personality-aware weighted techniques
- Provides rich reasoning that highlights diverse personality perspectives

#### Ensemble Methods (forecasting_tools/forecast_helpers/ensemble_methods.py)
- Implements various ensemble weighting strategies:
  - Equal weighting (baseline)
  - Performance-based weighting with recency decay
  - Personality diversity-maximizing weighting
  - Domain-specific personality weighting
  - Adaptive weighting that learns over time
- Provides specialized aggregation functions for different question types

### 2. Personality Diversity Management

#### Diversity Scoring (forecasting_tools/personality_management/diversity.py)
- Calculates diversity metrics across personality ensembles
- Identifies coverage gaps in personality trait space
- Recommends personality additions to maximize diversity
- Provides visualization tools for personality distribution analysis

### 3. Competition Framework

#### Competition Tracking (forecasting_tools/forecast_helpers/competition.py)
- Tracks performance metrics by bot type, personality, and domain
- Evaluates personality effectiveness across different question types
- Generates rich performance analytics and visualizations
- Supports saving and loading competition results

#### Personality Tournament (forecasting_tools/scripts/personality_tournament.py)
- Runs comparative tournaments across personalities and question domains
- Analyzes personality performance patterns
- Generates comprehensive reports and visualizations
- Supports incremental/resumable tournament execution

### 4. Adaptive Personality System

#### Personality Adaptation (forecasting_tools/personality_management/adaptation.py)
- Dynamically adjusts personality traits based on forecast performance
- Implements feedback loops for continuous performance improvement
- Creates hybrid personalities optimized for specific domains
- Provides mechanisms for trait combination and optimization

## Integration with Existing Framework

The personality management system has been integrated at multiple levels:

1. **Bot Level**: Each bot type (Basic, Research, Calibrated, Economist, Bayesian) now accepts and leverages personality traits to customize their forecasting approach.

2. **Ensemble Level**: The ensemble system creates diverse teams of forecasters with complementary personality traits, optimizing for both diversity and performance.

3. **Competition Level**: Performance tracking now includes personality dimensions, enabling analysis of which personalities perform best in which domains.

4. **Adaptation Level**: The system learns from performance data to optimize personality traits for specific domains and question types.

## Key Capabilities

### 1. Personality-Diverse Ensembles
- Creates ensembles with complementary personality traits
- Balances diversity with performance considerations
- Adapts ensemble composition based on question domain

### 2. Domain-Optimized Personalities
- Identifies which personalities perform best in each domain
- Creates hybrid personalities optimized for specific domains
- Dynamically adjusts traits based on performance feedback

### 3. Performance Analysis
- Tracks performance by personality across domains
- Identifies personality strengths and weaknesses
- Generates insights into personality-domain relationships

### 4. Continuous Improvement
- Learns from past performance to improve forecasts
- Adapts personality traits based on feedback
- Creates increasingly specialized personalities for each domain

## Usage Examples

### Creating a Personality-Diverse Ensemble
```python
from forecasting_tools.forecast_bots.ensemble_bot import EnsembleBot

# Create an ensemble with diverse personalities
ensemble = EnsembleBot(
    personality_names=["balanced", "bayesian", "economist", "creative", "cautious"],
    bot_types=["basic", "research", "calibrated"]
)

# Make a forecast
report = await ensemble.forecast_question(question)
```

### Running a Personality Tournament
```bash
python forecasting-tools/scripts/personality_tournament.py \
  --personalities balanced,bayesian,economist,creative,cautious \
  --bot-types basic,research,calibrated \
  --ai-competition \
  --output-dir tournament_results
```

### Using the Adaptation System
```python
from forecasting_tools.personality_management.adaptation import PersonalityFeedbackLoop
from forecasting_tools.personality_management import PersonalityManager

# Load base personalities
pm = PersonalityManager()
base_personalities = {
    "balanced": pm.load_personality("balanced"),
    "bayesian": pm.load_personality("bayesian"),
    "economist": pm.load_personality("economist")
}

# Create feedback loop
feedback_loop = PersonalityFeedbackLoop(
    base_personality_configs=base_personalities,
    save_dir="adapted_personalities"
)

# Record performance and adapt
feedback_loop.record_performance("balanced", "economics", 0.8)
feedback_loop.record_performance("bayesian", "economics", 0.6)

# Get optimal personality for a domain
optimal = feedback_loop.get_optimal_personality("economics")
``` 