#!/usr/bin/env python3
"""
Setup script for the personality management system.

This script:
1. Ensures all required directories exist
2. Installs required packages
3. Verifies that everything is set up correctly
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Required directories
REQUIRED_DIRS = [
    "forecasting-tools/forecasting_tools/personality_management",
    "forecasting-tools/forecasting_tools/personality_management/personalities",
    "forecasting-tools/forecasting_tools/personality_management/templates",
    "forecasting-tools/forecasting_tools/personality_management/utils",
]

# Required packages
REQUIRED_PACKAGES = [
    "pyyaml",
    "python-dotenv",
]

def create_dirs():
    """Create all required directories."""
    print("Creating directories...")
    for dir_path in REQUIRED_DIRS:
        path = Path(dir_path)
        if not path.exists():
            print(f"  Creating {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  {path} already exists")

def install_packages():
    """Install required packages."""
    print("\nInstalling required packages...")
    for package in REQUIRED_PACKAGES:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def verify_setup():
    """Verify that everything is set up correctly."""
    print("\nVerifying setup...")
    
    # Check directories
    all_dirs_exist = True
    for dir_path in REQUIRED_DIRS:
        path = Path(dir_path)
        if not path.exists():
            print(f"  ERROR: {path} does not exist")
            all_dirs_exist = False
    
    if all_dirs_exist:
        print("  All directories exist ✓")
    
    # Check packages
    all_packages_installed = True
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
            print(f"  {package} is installed ✓")
        except ImportError:
            print(f"  ERROR: {package} is not installed")
            all_packages_installed = False
    
    # Check for __init__.py files
    for dir_path in REQUIRED_DIRS:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            print(f"  Creating {init_file}")
            with open(init_file, "w") as f:
                f.write("# Auto-generated by setup script\n")
    
    return all_dirs_exist and all_packages_installed

def main():
    """Main function."""
    print("Setting up the personality management system...")
    create_dirs()
    install_packages()
    
    if verify_setup():
        print("\nSetup completed successfully! ✓")
        print("You can now use the personality management system.")
    else:
        print("\nSetup completed with errors. Please fix them manually.")

if __name__ == "__main__":
    main() 