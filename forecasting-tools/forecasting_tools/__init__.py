"""
Forecasting tools package for forecasting bots
"""

import nest_asyncio

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateResearcher as BaseRateResearcher,
)
from forecasting_tools.agents_and_tools.base_rates.estimator import (
    Estimator as Estimator,
)
from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    FactCheckedItem as FactCheckedItem,
)
from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    NicheListResearcher as NicheListResearcher,
)
from forecasting_tools.agents_and_tools.key_factors_researcher import (
    KeyFactorsResearcher as KeyFactorsResearcher,
)
from forecasting_tools.agents_and_tools.key_factors_researcher import (
    ScoredKeyFactor as ScoredKeyFactor,
)
from forecasting_tools.agents_and_tools.question_generators.question_generator import (
    QuestionGenerator as QuestionGenerator,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator as TopicGenerator,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import (
    clean_indents as clean_indents,
)
from forecasting_tools.ai_models.deprecated_model_classes.claude35sonnet import (
    Claude35Sonnet as Claude35Sonnet,
)
from forecasting_tools.ai_models.deprecated_model_classes.deepseek_r1 import (
    DeepSeekR1 as DeepSeekR1,
)
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import (
    Gpt4o as Gpt4o,
)
from forecasting_tools.ai_models.deprecated_model_classes.gpt4ovision import (
    Gpt4oVision as Gpt4oVision,
)
from forecasting_tools.ai_models.deprecated_model_classes.metaculus4o import (
    Gpt4oMetaculusProxy as Gpt4oMetaculusProxy,
)
from forecasting_tools.ai_models.deprecated_model_classes.perplexity import (
    Perplexity as Perplexity,
)
from forecasting_tools.ai_models.exa_searcher import ExaSearcher as ExaSearcher
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager as MonetaryCostManager,
)
from forecasting_tools.data_models.benchmark_for_bot import (
    BenchmarkForBot as BenchmarkForBot,
)
from forecasting_tools.data_models.binary_report import (
    BinaryReport as BinaryReport,
)
from forecasting_tools.data_models.data_organizer import (
    DataOrganizer as DataOrganizer,
)
from forecasting_tools.data_models.forecast_report import (
    ForecastReport as ForecastReport,
)
from forecasting_tools.data_models.forecast_report import (
    ReasonedPrediction as ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport as MultipleChoiceReport,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption as PredictedOption,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList as PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution as NumericDistribution,
)
from forecasting_tools.data_models.numeric_report import (
    NumericReport as NumericReport,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion as BinaryQuestion,
)
from forecasting_tools.data_models.questions import (
    MetaculusQuestion as MetaculusQuestion,
)
from forecasting_tools.data_models.questions import (
    MultipleChoiceQuestion as MultipleChoiceQuestion,
)
from forecasting_tools.data_models.questions import (
    NumericQuestion as NumericQuestion,
)
from forecasting_tools.data_models.questions import (
    QuestionState as QuestionState,
)
from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBot as EnhancedTemplateBot,
)
from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBotQ1 as EnhancedTemplateBotQ1,
)
from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBotQ2 as EnhancedTemplateBotQ2,
)
from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBotQ3 as EnhancedTemplateBotQ3,
)
from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBotQ4 as EnhancedTemplateBotQ4,
)
from forecasting_tools.forecast_bots.forecast_bot import (
    ForecastBot as ForecastBot,
)
from forecasting_tools.forecast_bots.forecast_bot import Notepad as Notepad
from forecasting_tools.forecast_bots.main_bot import MainBot as MainBot
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025 as Q1TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025 as Q2TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024 as Q3TemplateBot2024,
)
from forecasting_tools.forecast_bots.official_bots.q4_template_bot import (
    Q4TemplateBot2024 as Q4TemplateBot2024,
)
from forecasting_tools.forecast_bots.other.uniform_probability_bot import (
    UniformProbabilityBot as UniformProbabilityBot,
)
from forecasting_tools.forecast_bots.template_bot import (
    TemplateBot as TemplateBot,
)
from forecasting_tools.forecast_helpers.perplexity_searcher import (
    PerplexitySearcher as PerplexitySearcher,
)
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor as PredictionExtractor,
)
from forecasting_tools.forecast_helpers.smart_searcher import (
    SmartSearcher as SmartSearcher,
)
from forecasting_tools.personality_templates import (
    PersonalityConfig as PersonalityConfig,
)
from forecasting_tools.personality_templates import (
    PersonalityManager as PersonalityManager,
)
from forecasting_tools.personality_templates import (
    TemplateManager as TemplateManager,
)

nest_asyncio.apply()
