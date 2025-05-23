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

logger = logging.getLogger(__name__)


class ForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion
    personality_name: str = "balanced"


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
        # st.write(
        #     "Enter the information for your question. Exa.ai is used to gather up to date information. Each citation attempts to link to a highlight of the a ~4 sentence quote found with Exa.ai. This project is in beta some inaccuracies are expected."
        # )
        pass

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
                    personality_name=personality_name
                )
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        with st.spinner(f"Forecasting with {input.personality_name} personality... This may take a minute or two..."):
            # Create a bot with the selected personality
            bot = MainBot(
                research_reports_per_question=1,
                predictions_per_research_report=5,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
                personality_name=input.personality_name
            )
            
            # Display personality preview
            with st.expander(f"Using {input.personality_name} personality", expanded=True):
                # Get personality description
                try:
                    personality_manager = PersonalityManager()
                    personality = personality_manager.load_personality(input.personality_name)
                    
                    # Display basic info about how this personality approaches forecasting
                    st.markdown(f"### {personality.name}")
                    if personality.description:
                        st.markdown(f"*{personality.description}*")
                    
                    # Display personality traits summary
                    st.markdown("#### Forecasting Approach")
                    st.markdown(f"- **Thinking Style:** {personality.thinking_style.value}")
                    st.markdown(f"- **Uncertainty Approach:** {personality.uncertainty_approach.value}")
                    st.markdown(f"- **Reasoning Depth:** {personality.reasoning_depth.value}")
                    
                    # Create a simple progress bar to show the forecasting is active
                    progress_bar = st.progress(0)
                    for i in range(5):
                        # Simulate activity while forecasting is happening
                        progress_bar.progress((i + 1) * 20)
                        
                except Exception as e:
                    st.warning(f"Could not load personality details: {str(e)}")
            
            # Generate forecast
            report = await bot.forecast_question(input.question)
            assert isinstance(report, BinaryReport)
            
            # Add personality info to report
            report.metadata = report.metadata or {}
            report.metadata["personality_name"] = input.personality_name
            
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
            # Get personality name from metadata if available
            personality_name = "balanced"
            if hasattr(output, "metadata") and output.metadata:
                personality_name = output.metadata.get("personality_name", "balanced")
            
            # Add personality name to the report display
            st.markdown(f"### Forecast by {personality_name.capitalize()} Personality")
        
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
