import os
import sys

import dotenv
import streamlit as st

from forecasting_tools.front_end.app_pages.benchmark_page import BenchmarkPage
from forecasting_tools.front_end.app_pages.chat_page import ChatPage

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)


from forecasting_tools.front_end.app_pages.base_rate_page import BaseRatePage
from forecasting_tools.front_end.app_pages.estimator_page import EstimatorPage
from forecasting_tools.front_end.app_pages.forecaster_page import (
    ForecasterPage,
)
from forecasting_tools.front_end.app_pages.key_factors_page import (
    KeyFactorsPage,
)
from forecasting_tools.front_end.app_pages.niche_list_researcher_page import (
    NicheListResearchPage,
)
from forecasting_tools.front_end.app_pages.personality_manager_page import (
    PersonalityManagerPage,
)
from forecasting_tools.front_end.app_pages.personality_analytics_page import (
    PersonalityAnalyticsPage,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.util.custom_logger import CustomLogger


class HomePage(AppPage):
    PAGE_DISPLAY_NAME: str = "🏠 Home"
    URL_PATH: str = "/"
    IS_DEFAULT_PAGE: bool = True

    CHAT_PAGE: type[AppPage] = ChatPage
    FORECASTER_PAGE: type[AppPage] = ForecasterPage
    BASE_RATE_PAGE: type[AppPage] = BaseRatePage
    NICHE_LIST_RESEARCH_PAGE: type[AppPage] = NicheListResearchPage
    ESTIMATOR_PAGE: type[AppPage] = EstimatorPage
    KEY_FACTORS_PAGE: type[AppPage] = KeyFactorsPage
    BENCHMARK_PAGE: type[AppPage] = BenchmarkPage
    PERSONALITY_MANAGER_PAGE: type[AppPage] = PersonalityManagerPage
    PERSONALITY_ANALYTICS_PAGE: type[AppPage] = PersonalityAnalyticsPage
    
    NON_HOME_PAGES: list[type[AppPage]] = [
        CHAT_PAGE,
        FORECASTER_PAGE,
        KEY_FACTORS_PAGE,
        BASE_RATE_PAGE,
        NICHE_LIST_RESEARCH_PAGE,
        ESTIMATOR_PAGE,
        PERSONALITY_MANAGER_PAGE,
        PERSONALITY_ANALYTICS_PAGE,
    ]

    @classmethod
    async def _async_main(cls) -> None:
        st.title("Forecasting Tools")

        st.markdown(
            """
            Welcome to the Forecasting Tools suite! This application provides a range of forecasting capabilities using AI.
            
            ## Available Tools
            
            ### Core Forecasting
            - **🔍 Forecast a Question**: Generate forecasts for binary questions
            - **📊 Key Factors Analysis**: Identify and analyze key factors that influence a forecast
            - **📈 Base Rate Finder**: Find relevant base rates for your forecasts
            - **🧮 Estimator**: Generate estimates for numerical questions
            
            ### Personality Management
            - **👤 Personality Manager**: Create, edit, and manage forecaster personalities
            - **📊 Personality Analytics**: Analyze personality performance across domains
            
            ### Research Tools
            - **🔎 Niche List Researcher**: Generate comprehensive lists for research questions
            - **📚 Benchmark Questions**: Explore benchmark questions for testing
            
            ## Getting Started
            
            Select a tool from the sidebar to begin. Each tool provides specific forecasting capabilities.
            
            For questions or feedback, please contact the development team.
            """
        )

        # Add sidebar with links to different tools
        st.sidebar.title("Navigation")

        st.sidebar.header("Core Forecasting")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/forecaster_page.py", label="🔍 Forecast a Question")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/key_factors_page.py", label="📊 Key Factors Analysis")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/base_rate_page.py", label="📈 Base Rate Finder")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/estimator_page.py", label="🧮 Estimator")

        st.sidebar.header("Personality Management")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/personality_manager_page.py", label="👤 Personality Manager")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/personality_analytics_page.py", label="📊 Personality Analytics")

        st.sidebar.header("Research Tools")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/niche_list_researcher_page.py", label="🔎 Niche List Researcher")
        st.sidebar.page_link("forecasting_tools/front_end/app_pages/benchmark_page.py", label="📚 Benchmark Questions")

        for page in cls.NON_HOME_PAGES:
            label = page.PAGE_DISPLAY_NAME
            if st.button(label, key=label):
                st.switch_page(page.convert_to_streamlit_page())


def run_forecasting_streamlit_app() -> None:
    all_pages = [HomePage] + HomePage.NON_HOME_PAGES
    if os.getenv("LOCAL_STREAMLIT_MODE", "false").lower() == "true":
        all_pages.append(HomePage.BENCHMARK_PAGE)
    navigation = st.navigation(
        [page.convert_to_streamlit_page() for page in all_pages]
    )
    st.set_page_config(
        page_title="Forecasting-Tools", page_icon="🔮", layout="wide", initial_sidebar_state="expanded"
    )
    navigation.run()


if __name__ == "__main__":
    dotenv.load_dotenv()
    if "logger_initialized" not in st.session_state:
        CustomLogger.clear_latest_log_files()
        CustomLogger.setup_logging()
        st.session_state["logger_initialized"] = True
    run_forecasting_streamlit_app()
