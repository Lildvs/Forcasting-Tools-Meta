"""Streamlit Cloud entry point.

This file redirects to the main application entry point in front_end/Home.py.
"""

import os
import sys
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import and run the main app
try:
    from front_end.Home import main
    logging.info("Successfully imported main function from front_end/Home.py")
except ImportError as e:
    logging.error(f"Failed to import from front_end/Home.py: {e}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        logging.info("Starting application...")
        main()
    except Exception as e:
        logging.error(f"Error running application: {e}")
        raise 