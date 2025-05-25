"""
Streamlit Cloud entry point.

This file redirects to the main application entry point in front_end/Home.py.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the main function
from front_end.Home import main

# Run the application
if __name__ == "__main__":
    logging.info("Starting application...")
    main() 