#!/usr/bin/env python
"""
Run the Streamlit app.

This script provides a convenient way to run the Streamlit app from Python.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit app."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Create data directory if it doesn't exist
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Path to the Streamlit app
    app_path = script_dir / "streamlit_app.py"
    
    # Check if the app exists
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        return 1
    
    # Get command line arguments
    args = sys.argv[1:]
    
    # Run the Streamlit app
    cmd = ["streamlit", "run", str(app_path)] + args
    
    print(f"Running Streamlit app: {' '.join(cmd)}")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 