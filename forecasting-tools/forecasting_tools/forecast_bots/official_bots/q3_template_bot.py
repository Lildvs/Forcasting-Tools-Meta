from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.deprecated_model_classes.perplexity import (
    Perplexity,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)


class Q3TemplateBot2024(ForecastBot):
    """
    This is the template bot for the Q3 2024 Metaculus AI Tournament.
    It should be exactly the same except for Perplexity running on a new model (the original model was deprecated)
    Find the q3 bot here: https://github.com/Metaculus/metac-bot/commit/e459f2958f66658783057da46e257896b49607be
    This comment was last updated on Jan 20 2025
    """

    FINAL_DECISION_LLM = GeneralLlm(
        model="gpt-4.1", temperature=0.1
    )  # Q3 Bot used the default llama index temperature which as of Dec 21 2024 is 0.1

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": cls.FINAL_DECISION_LLM,
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        # Get the research prompt from the personality manager
        system_prompt = self.personality_manager.get_prompt("research_prompt")

        # Note: The original q3 bot did not set temperature, and I could not find the default temperature of perplexity
        response = await Perplexity(
            temperature=0.1, system_prompt=system_prompt
        ).invoke(question.question_text)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Get the binary forecast prompt from the personality manager
        prompt = self.personality_manager.get_prompt(
            "binary_forecast_prompt",
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Apply thinking configuration if available
        thinking_config = self.personality_manager.get_thinking_config()
        
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, 
            **thinking_config if thinking_config else {}
        )
        
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raise NotImplementedError("Multiple choice was not supported in Q3")

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError("Numeric was not supported in Q3")
