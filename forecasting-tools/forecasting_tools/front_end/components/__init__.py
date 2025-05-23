"""
UI Components for Forecasting Tools

This package provides reusable UI components for visualizing forecasts,
displaying research sources, reasoning steps, and confidence information.
"""

from forecasting_tools.front_end.components.probability_visualization import (
    display_binary_forecast,
    display_numeric_forecast,
    create_binary_probability_chart,
    create_numeric_distribution_chart,
    create_probability_timeline
)

from forecasting_tools.front_end.components.research_sources import (
    display_evidence_item,
    create_evidence_from_text,
    extract_evidence_from_markdown,
    display_research_sources,
    display_reasoning_steps,
    display_biases_and_uncertainties
)

from forecasting_tools.front_end.components.confidence_display import (
    create_confidence_gauge,
    display_confidence_breakdown,
    display_calibration_info,
    create_confidence_interval_display
)

from forecasting_tools.front_end.components.ui_utils import (
    load_css,
    create_card,
    display_data_table,
    display_info_box,
    download_button,
    create_tabs_container,
    set_page_container_style,
    loader_animation,
    show_notification
)

__all__ = [
    # Probability visualization
    'display_binary_forecast',
    'display_numeric_forecast',
    'create_binary_probability_chart',
    'create_numeric_distribution_chart',
    'create_probability_timeline',
    
    # Research sources
    'display_evidence_item',
    'create_evidence_from_text',
    'extract_evidence_from_markdown',
    'display_research_sources',
    'display_reasoning_steps',
    'display_biases_and_uncertainties',
    
    # Confidence display
    'create_confidence_gauge',
    'display_confidence_breakdown',
    'display_calibration_info',
    'create_confidence_interval_display',
    
    # UI utilities
    'load_css',
    'create_card',
    'display_data_table',
    'display_info_box',
    'download_button',
    'create_tabs_container',
    'set_page_container_style',
    'loader_animation',
    'show_notification'
] 