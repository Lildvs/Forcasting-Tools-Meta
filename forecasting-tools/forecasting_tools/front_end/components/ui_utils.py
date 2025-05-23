import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import os
from pathlib import Path
import base64

def load_css(css_file: Optional[str] = None, css_string: Optional[str] = None):
    """
    Load CSS styling into the Streamlit app.
    
    Args:
        css_file: Path to CSS file
        css_string: CSS code as string
    """
    if css_file and os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    elif css_string:
        st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)

def create_card(
    title: str,
    content: str,
    footer: Optional[str] = None,
    icon: Optional[str] = None,
    is_expanded: bool = False
):
    """
    Create a card-like UI element.
    
    Args:
        title: Card title
        content: Card content (can be markdown)
        footer: Optional footer text
        icon: Optional icon emoji
        is_expanded: Whether to expand the card by default
    """
    icon_html = f"<span style='margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    
    with st.expander(f"{icon_html}{title}", expanded=is_expanded):
        st.markdown(content, unsafe_allow_html=True)
        
        if footer:
            st.markdown(f"<div style='font-size: 0.8rem; color: #58595b; border-top: 1px solid #e6e9ef; margin-top: 1rem; padding-top: 0.5rem;'>{footer}</div>", unsafe_allow_html=True)

def display_data_table(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    columns: Optional[List[str]] = None,
    column_formatters: Optional[Dict[str, Callable]] = None,
    max_rows: Optional[int] = None,
    height: Optional[int] = None,
    use_pagination: bool = True
):
    """
    Display data in a table with formatting.
    
    Args:
        data: Data as list of dicts or DataFrame
        columns: Optional list of columns to display
        column_formatters: Optional formatters for column values
        max_rows: Maximum rows to display
        height: Optional table height
        use_pagination: Whether to use pagination
    """
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Filter columns if specified
    if columns:
        df = df[columns]
    
    # Apply formatters
    if column_formatters:
        for col, formatter in column_formatters.items():
            if col in df.columns:
                df[col] = df[col].apply(formatter)
    
    # Truncate if max_rows specified
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    # Display with formatting
    if use_pagination:
        st.dataframe(df, use_container_width=True, height=height)
    else:
        st.table(df)

def display_info_box(
    message: str,
    type: str = "info", 
    icon: Optional[str] = None,
    dismissible: bool = False
):
    """
    Display an information box.
    
    Args:
        message: Message to display
        type: Box type (info, success, warning, error)
        icon: Optional icon override
        dismissible: Whether the box is dismissible
    """
    # Icons for different box types
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    # Colors for different box types
    colors = {
        "info": "#cce5ff",
        "success": "#d4edda",
        "warning": "#fff3cd",
        "error": "#f8d7da"
    }
    
    # Border colors
    border_colors = {
        "info": "#b8daff",
        "success": "#c3e6cb",
        "warning": "#ffeeba",
        "error": "#f5c6cb"
    }
    
    # Use specified icon or default for the type
    display_icon = icon if icon else icons.get(type, icons["info"])
    
    # Get colors for the specified type
    bg_color = colors.get(type, colors["info"])
    border_color = border_colors.get(type, border_colors["info"])
    
    # Create dismissible functionality
    dismiss_js = """
    <script>
    const closeButtons = document.getElementsByClassName('info-box-close');
    for (let i = 0; i < closeButtons.length; i++) {
        closeButtons[i].addEventListener('click', function() {
            this.parentElement.style.display = 'none';
        });
    }
    </script>
    """
    
    dismiss_button = """<span class="info-box-close" style="float: right; cursor: pointer; font-weight: bold;">×</span>""" if dismissible else ""
    
    # Create the info box HTML
    html = f"""
    <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;">
        {dismiss_button}
        <span style="font-weight: bold; margin-right: 0.5rem;">{display_icon}</span>
        {message}
    </div>
    {dismiss_js if dismissible else ""}
    """
    
    st.markdown(html, unsafe_allow_html=True)

def download_button(
    data: Any,
    file_name: str,
    button_text: str = "Download",
    mime: str = "text/plain"
):
    """
    Create a download button for any data.
    
    Args:
        data: Data to download
        file_name: Name of the file to download
        button_text: Text to display on the button
        mime: MIME type of the file
    """
    # Convert data to appropriate format
    if isinstance(data, pd.DataFrame):
        if mime == "text/csv":
            data = data.to_csv(index=False)
        elif mime == "application/json":
            data = data.to_json(orient="records")
        else:
            data = data.to_string()
    elif not isinstance(data, str):
        data = str(data)
    
    # Create the download button
    b64 = base64.b64encode(data.encode()).decode()
    button_html = f"""
    <a href="data:{mime};base64,{b64}" download="{file_name}">
        <button style="
            background-color: #0068c9;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        ">
            <span style="margin-right: 0.5rem;">📥</span> {button_text}
        </button>
    </a>
    """
    st.markdown(button_html, unsafe_allow_html=True)

def create_tabs_container():
    """
    Create a custom tabs container with styling.
    
    Returns:
        Tuple of (tab_bar, tab_content)
    """
    # Create container for tab bar
    tab_bar = st.container()
    
    # Create container for tab content
    tab_content = st.container()
    
    return tab_bar, tab_content

def set_page_container_style(
    max_width: int = 1200,
    padding_top: int = 2,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
    color_bg: str = "#f0f2f6",
    color_fg: str = "#262730"
):
    """
    Set page container style.
    
    Args:
        max_width: Maximum width of the container
        padding_top: Top padding in rem
        padding_right: Right padding in rem
        padding_left: Left padding in rem
        padding_bottom: Bottom padding in rem
        color_bg: Background color
        color_fg: Foreground color
    """
    style = f"""
    <style>
        .reportview-container .main .block-container {{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {color_fg};
            background-color: {color_bg};
        }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def loader_animation():
    """Display a loading animation."""
    return st.markdown(
        """
        <style>
        .loader {
            border: 8px solid #f3f3f3;
            border-radius: 50%;
            border-top: 8px solid #0068c9;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        <div class="loader"></div>
        """, 
        unsafe_allow_html=True
    )

def show_notification(message: str, type: str = "info", duration: int = 3):
    """
    Show a temporary notification.
    
    Args:
        message: Message to display
        type: Notification type (info, success, warning, error)
        duration: Duration in seconds
    """
    # Colors for different notification types
    colors = {
        "info": "#0068c9",
        "success": "#09ab3b",
        "warning": "#ff9800",
        "error": "#ff4b4b"
    }
    
    # Background colors
    bg_colors = {
        "info": "#e6f0ff",
        "success": "#e6f8e6",
        "warning": "#fff8e6",
        "error": "#ffe6e6"
    }
    
    # Get color for the specified type
    color = colors.get(type, colors["info"])
    bg_color = bg_colors.get(type, bg_colors["info"])
    
    # Create a unique key for this notification
    import uuid
    key = str(uuid.uuid4())
    
    # Create the notification HTML with fade-out effect
    st.markdown(
        f"""
        <style>
        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            75% {{ opacity: 1; }}
            100% {{ opacity: 0; display: none; }}
        }}
        .notification-{key} {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: {bg_color};
            color: {color};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {color};
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            z-index: 9999;
            animation: fadeOut {duration}s forwards;
        }}
        </style>
        <div class="notification-{key}">
            {message}
        </div>
        """, 
        unsafe_allow_html=True
    ) 