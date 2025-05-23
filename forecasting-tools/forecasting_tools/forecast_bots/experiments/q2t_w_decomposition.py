import asyncio
import logging
import os
from datetime import datetime
from typing import Literal, cast

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
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
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher

logger = logging.getLogger(__name__)


class Q2TemplateWithDecompositionBot(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament, with added decomposition.
    This is an experimental bot to test decomposition.
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Run the research for the question"""
        research = await self._run_research(question)
        return research

    async def _run_research(self, question: MetaculusQuestion) -> str:
        try:
            # First get information from search
            perplexity_research = await PerplexitySearcher().get_formatted_news_async(
                question.question_text
            )

            # Then run decomposition
            prompt = clean_indents(
                f"""
                You are an expert forecaster. You're being asked to forecast the answer to the question: "{question.question_text}"

                TASK: Generate 4-5 simpler sub-questions that would help to answer this question.

                Context: {question.background_info if hasattr(question, "background_info") else ""}
                """
            )
            llm = GeneralLlm(model="openai/gpt-4o-mini", temperature=0.2)
            decomposition = await llm.invoke(prompt)

            # For each subquestion, we want to get info
            prompt = clean_indents(
                f"""
                You are an expert researcher. Your task is to generate highly relevant information to help answer questions.

                Question: {question.question_text}

                You've broken down the question into these sub-questions:
                {decomposition}

                Now, for each of these sub-questions, answer it with the most relevant information available. Provide factual information with precise numbers and dates where available. Cite sources if possible.
                """
            )
            sub_answers = await llm.invoke(prompt)

            # Combine information in a useful way
            combined_research = f"""
            # PERPLEXITY SEARCH RESULTS
            {perplexity_research}

            # QUESTION DECOMPOSITION
            {decomposition}

            # SUB-QUESTION ANSWERS
            {sub_answers}
            """

            return combined_research
        except Exception as e:
            logger.error(f"Error in run_research: {e}")
            return ""

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """Run the forecast for the binary question"""
        llm_choice = self.get_llm("default", "llm")

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant has provided you with the following information:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await llm_choice.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """Default LLM config for the bot"""
        return {
            "default": "openai/gpt-4o-mini",
            "researcher": "perplexity/news-summaries",
        }


class Q2TemplateBotWithDecompositionV1(Q2TemplateBot2025):
    """
    Runs forecasts on decomposed sub questions separately
    """

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        gemini_grounded_model = GeneralLlm.grounded_model(
            model="openrouter/google/gemini-2.5-pro-preview",
            temperature=0.3,
        )
        gemini_model = GeneralLlm(
            model="openrouter/google/gemini-2.5-pro-preview",
            temperature=0.3,
        )
        return {
            "default": gemini_model,
            "summarizer": "gpt-4o",
            "decomposer": gemini_grounded_model,
            "researcher": "perplexity/news-summaries",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        perplexity_research = await PerplexitySearcher().get_formatted_news_async(
            question.question_text
        )

        question_context = clean_indents(
            f"""
            Here are more details for the original question:

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )

        model = self.get_llm("decomposer", "llm")
        decomposition_result = (
            await QuestionDecomposer().decompose_into_questions_deep(
                model=model,
                fuzzy_topic_or_question=question.question_text,
                number_of_questions=5,
                related_research=perplexity_research,
                additional_context=question_context,
            )
        )
        logger.info(f"Decomposition result: {decomposition_result}")

        operationalize_tasks = [
            QuestionOperationalizer(model=model).operationalize_question(
                question_title=question,
                related_research=None,
            )
            for question in decomposition_result.questions
        ]
        operationalized_questions = await asyncio.gather(*operationalize_tasks)

        metaculus_questions = (
            SimpleQuestion.simple_questions_to_metaculus_questions(
                operationalized_questions
            )
        )
        sub_predictor = Q2TemplateBot2025(
            llms=self._llms,
            predictions_per_research_report=5,
            research_reports_per_question=1,
        )
        forecasts = await sub_predictor.forecast_questions(
            metaculus_questions, return_exceptions=True
        )

        formatted_forecasts = ""
        sub_question_bullets = ""
        for forecast in forecasts:
            if isinstance(forecast, BaseException):
                logger.error(f"Error forecasting on question: {forecast}")
                continue
            formatted_forecasts += (
                f"QUESTION: {forecast.question.question_text}\n\n"
            )
            formatted_forecasts += f"PREDICTION: {forecast.make_readable_prediction(forecast.prediction)}\n\n"
            formatted_forecasts += f"SUMMARY: {forecast.summary}\n\n----------------------------------------\n\n"
            sub_question_bullets += f"- {forecast.question.question_text}\n"
        research = clean_indents(
            f"""
            ==================== NEWS ====================

            {perplexity_research}

            ==================== FORECAST HISTORY ====================

            Below are some related questions you have forecasted on before:

            {sub_question_bullets if sub_question_bullets else "<No related questions>"}

            Below are some forecasts you have made on these question:

            {formatted_forecasts if formatted_forecasts else "<No previous forecasts>"}

            ==================== END ====================
            """
        )
        logger.info(research)
        return research


class Q2TemplateBotWithDecompositionV2(Q2TemplateBot2025):
    """
    Runs forecasts on all decomposed questions simultaneously
    """

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": "openai/gpt-4o-mini",
            "researcher": "perplexity/news-summaries",
            "decomposer": "openai/gpt-4o-mini",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")

            # Just use perplexity no matter what the researcher is set to
            perplexity_research = await PerplexitySearcher().get_formatted_news_async(
                question.question_text
            )

            # Then run decomposition
            prompt = clean_indents(
                f"""
                You are an expert forecaster. You're being asked to forecast the answer to the question: "{question.question_text}"

                TASK: Generate 4-5 simpler sub-questions that would help to answer this question.

                Context: {question.background_info if hasattr(question, "background_info") else ""}
                """
            )
            llm = self.get_llm("decomposer")
            decomposition = await llm.invoke(prompt)

            # For each subquestion, we want to get info
            prompt = clean_indents(
                f"""
                You are an expert researcher. Your task is to generate highly relevant information to help answer questions.

                Question: {question.question_text}

                You've broken down the question into these sub-questions:
                {decomposition}

                Now, for each of these sub-questions, answer it with the most relevant information available. Provide factual information with precise numbers and dates where available. Cite sources if possible.
                """
            )
            sub_answers = await llm.invoke(prompt)

            # Combine information in a useful way
            combined_research = f"""
            # PERPLEXITY SEARCH RESULTS
            {perplexity_research}

            # QUESTION DECOMPOSITION
            {decomposition}

            # SUB-QUESTION ANSWERS
            {sub_answers}
            """

        return combined_research

    async def _get_sub_questions_as_bullets(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        scratch_pad = await self._get_notepad(question)
        sub_questions = scratch_pad.note_entries[research]
        formatted_sub_questions = ""
        for sub_question in sub_questions:
            formatted_sub_questions += f"- {sub_question}\n"
        return formatted_sub_questions

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        formatted_sub_questions = await self._get_sub_questions_as_bullets(
            question, research
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            (e) Your forecasts for all the following sub-questions
            {formatted_sub_questions}

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        formatted_sub_questions = await self._get_sub_questions_as_bullets(
            question, research
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.
            (d) Your forecasts for all the following sub-questions
            {formatted_sub_questions}

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        formatted_sub_questions = await self._get_sub_questions_as_bullets(
            question, research
        )
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) Your forecasts for all the following sub-questions
            {formatted_sub_questions}

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
