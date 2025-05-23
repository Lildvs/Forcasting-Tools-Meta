import logging
import re

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.front_end.helpers.personality_selector import PersonalitySelector
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.llm_config import LLMConfigManager
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.front_end.components.ui_utils import display_llm_workflow_diagram

logger = logging.getLogger(__name__)


class ForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion
    personality_name: str = "balanced"
    model_name: str = "gpt-4.1"  # Updated default to latest version


class ForecasterPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔍 Forecast a Question"
    URL_PATH: str = "/forecast"
    INPUT_TYPE = ForecastInput
    OUTPUT_TYPE = BinaryReport
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/forecast_page_examples.json"

    # Form input keys
    QUESTION_TEXT_BOX = "question_text_box"
    RESOLUTION_CRITERIA_BOX = "resolution_criteria_box"
    FINE_PRINT_BOX = "fine_print_box"
    BACKGROUND_INFO_BOX = "background_info_box"
    NUM_BACKGROUND_QUESTIONS_BOX = "num_background_questions_box"
    NUM_BASE_RATE_QUESTIONS_BOX = "num_base_rate_questions_box"
    METACULUS_URL_INPUT = "metaculus_url_input"
    FETCH_BUTTON = "fetch_button"
    PERSONALITY_SELECTION = "personality_selection"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # Display the multi-layered LLM workflow diagram
        st.subheader("Advanced Multi-Layered LLM Forecasting")
        st.markdown("""
        Our forecasting system uses a sophisticated multi-layered approach with specialized LLM models 
        for different stages of the forecasting process. This architecture combines the strengths of 
        different models to produce more accurate and well-researched forecasts.
        """)
        
        # Display the workflow diagram
        display_llm_workflow_diagram()
        
        # Add a divider before the rest of the content
        st.markdown("---")

    @classmethod
    async def _get_input(cls) -> ForecastInput | None:
        cls.__display_metaculus_url_input()
        
        # Add personality selection before the form
        st.subheader("Select Forecaster Personality")
        
        # Display domain selection for personality recommendation
        domain_options = [
            "General (No specific domain)",
            "Economics", "Finance", "Politics", "Technology", 
            "Science", "Health", "Sports", "Entertainment",
            "Geopolitics", "Environment", "Energy", "Social"
        ]
        
        selected_domain = st.selectbox(
            "Question Domain (for personality recommendation):",
            options=domain_options,
            index=0
        )
        
        # Get domain without the parentheses part
        domain = selected_domain.split(" (")[0] if "(" in selected_domain else selected_domain
        
        # Get optimal personality for domain if a specific domain is selected
        filter_by_domain = None
        if domain != "General":
            filter_by_domain = domain
            st.info(f"Based on the selected domain, we recommend personalities optimized for {domain} forecasting.")
        
        # Add model selection dropdown with explanation of the multi-layered approach
        st.subheader("Configure Model Layers")
        
        # Show explanation of the layered approach
        with st.expander("About the Multi-Layered Model Architecture", expanded=False):
            st.markdown("""
            Our forecasting system uses multiple specialized LLM models:
            
            1. **Base Layer** - The primary reasoning engine that analyzes the question and generates the final forecast
            2. **Researcher Layer** - Specialized model for gathering detailed information from external sources
            3. **Summarizer Layer** - Lightweight model for condensing research into key insights
            
            You can select the primary Base Layer model below. The other layers are configured automatically 
            for optimal performance.
            """)
        
        # Get models from LLMConfigManager
        config = LLMConfigManager.get_default_config()
        default_models = [
            "gpt-4.1",  # Latest default from config
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku"
        ]
        
        # Create columns for model selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "Select Primary (Base) Model:",
                options=default_models,
                index=0,
                help="Select the model to use as the primary reasoning engine. GPT-4.1 is recommended for best results."
            )
        
        with col2:
            # Show the current researcher model from config
            researcher_model = config["researcher"]["model"]
            st.info(f"Researcher Layer: {researcher_model}\nSummarizer Layer: {config['summarizer']['model']}")
        
        # Add personality selector
        selected_personality = PersonalitySelector.display_selector(
            key_prefix=cls.PERSONALITY_SELECTION,
            filter_by_domain=filter_by_domain
        )
        
        with st.form("forecast_form"):
            question_text = st.text_input(
                "Yes/No Binary Question", key=cls.QUESTION_TEXT_BOX
            )
            resolution_criteria = st.text_area(
                "Resolution Criteria (optional)",
                key=cls.RESOLUTION_CRITERIA_BOX,
            )
            fine_print = st.text_area(
                "Fine Print (optional)", key=cls.FINE_PRINT_BOX
            )
            background_info = st.text_area(
                "Background Info (optional)", key=cls.BACKGROUND_INFO_BOX
            )

            submitted = st.form_submit_button("Submit")

            if submitted:
                if not question_text:
                    st.error("Question Text is required.")
                    return None
                
                # Ensure we have a personality selected (use default if none)
                personality_name = selected_personality or "balanced"
                
                question = BinaryQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria=resolution_criteria,
                    fine_print=fine_print,
                    page_url="",
                    api_json={},
                )
                return ForecastInput(
                    question=question,
                    personality_name=personality_name,
                    model_name=selected_model
                )
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        with st.spinner(f"Forecasting with {input.personality_name} personality using {input.model_name}... This may take a minute or two..."):
            # Create a bot with the selected personality and model
            # Get default settings from LLMConfigManager
            config = LLMConfigManager.get_default_config()
            
            # Create custom LLM configuration
            llms = {
                "default": GeneralLlm(model=input.model_name, temperature=0.3),
                "summarizer": GeneralLlm(model=config["summarizer"]["model"], temperature=0.1),
                "researcher": GeneralLlm(model=config["researcher"]["model"], temperature=0.1),
            }
            
            # Create bot with custom configuration
            bot = MainBot(
                research_reports_per_question=1,
                predictions_per_research_report=5,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
                personality_name=input.personality_name,
                llms=llms
            )
            
            # Display process information with progress tracking
            st.markdown("### Forecast Generation in Progress")
            st.markdown("The system is working through the multi-layered LLM process:")
            
            # Create a progress indicator
            progress_container = st.empty()
            progress_bar = st.progress(0)
            
            # Display stage information
            stages = [
                "Stage 1: Initial analysis with Base LLM",
                "Stage 2: Research gathering with specialized Researcher LLM",
                "Stage 3: Research summarization with Summarizer LLM",
                "Stage 4: Final forecast generation with Base LLM"
            ]
            
            stage_container = st.empty()
            stage_container.info(stages[0])
            progress_bar.progress(10)
            
            # Simulate progress through the stages (actual progress tracking would require changes to the forecast process)
            import time
            import threading
            
            def update_progress():
                for i, stage in enumerate(stages):
                    stage_container.info(stage)
                    # Each stage takes approximately 25% of the total progress
                    start_progress = i * 25
                    for j in range(start_progress, start_progress + 25):
                        progress_bar.progress(j)
                        time.sleep(0.2)  # Adjust timing based on actual performance
            
            # Start progress update in a separate thread
            threading.Thread(target=update_progress).start()
            
            # Generate forecast
            report = await bot.forecast_question(input.question)
            assert isinstance(report, BinaryReport)
            
            # Complete the progress bar when done
            progress_bar.progress(100)
            stage_container.success("Forecast completed successfully!")
            
            # Add personality and model info to report
            report.metadata = report.metadata or {}
            report.metadata["personality_name"] = input.personality_name
            report.metadata["primary_model"] = input.model_name
            report.metadata["researcher_model"] = config["researcher"]["model"]
            report.metadata["summarizer_model"] = config["summarizer"]["model"]
            
            return report

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: ForecastInput,
        output: BinaryReport,
        is_premade: bool,
    ) -> None:
        if is_premade:
            output.price_estimate = 0
        ForecastDatabaseManager.add_forecast_report_to_database(
            output, run_type=ForecastRunType.WEB_APP_FORECAST
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[BinaryReport]) -> None:
        for output in outputs:
            # Get metadata from the report
            personality_name = "balanced"
            primary_model = "gpt-4.1"
            researcher_model = "Unknown"
            summarizer_model = "Unknown"
            
            if hasattr(output, "metadata") and output.metadata:
                personality_name = output.metadata.get("personality_name", personality_name)
                primary_model = output.metadata.get("primary_model", primary_model)
                researcher_model = output.metadata.get("researcher_model", researcher_model)
                summarizer_model = output.metadata.get("summarizer_model", summarizer_model)
            
            # Add model information to the report display
            st.markdown(f"### Forecast by {personality_name.capitalize()} Personality")
            st.markdown(f"**Models used:**")
            st.markdown(f"- Primary: {primary_model}")
            st.markdown(f"- Researcher: {researcher_model}")
            st.markdown(f"- Summarizer: {summarizer_model}")
        
        ReportDisplayer.display_report_list(outputs)

    @classmethod
    def __display_metaculus_url_input(cls) -> None:
        with st.expander("Use an existing Metaculus Binary question"):
            st.write(
                "Enter a Metaculus question URL to autofill the form below."
            )

            metaculus_url = st.text_input(
                "Metaculus Question URL", key=cls.METACULUS_URL_INPUT
            )
            fetch_button = st.button("Fetch Question", key=cls.FETCH_BUTTON)

            if fetch_button and metaculus_url:
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(metaculus_url)
                        metaculus_question = (
                            MetaculusApi.get_question_by_post_id(question_id)
                        )
                        if isinstance(metaculus_question, BinaryQuestion):
                            cls.__autofill_form(metaculus_question)
                        else:
                            st.error(
                                "Only binary questions are supported at this time."
                            )
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )

    @classmethod
    def __autofill_form(cls, question: BinaryQuestion) -> None:
        st.session_state[cls.QUESTION_TEXT_BOX] = question.question_text
        st.session_state[cls.BACKGROUND_INFO_BOX] = (
            question.background_info or ""
        )
        st.session_state[cls.RESOLUTION_CRITERIA_BOX] = (
            question.resolution_criteria or ""
        )
        st.session_state[cls.FINE_PRINT_BOX] = question.fine_print or ""


if __name__ == "__main__":
    dotenv.load_dotenv()
    ForecasterPage.main()
