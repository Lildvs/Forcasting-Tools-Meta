#!/usr/bin/env python3
"""
Generate requirements.txt files from Poetry's lock file.

This script extracts dependencies from Poetry's lock file and generates
various requirements.txt files for different environments.
"""

import os
import subprocess
import sys
from pathlib import Path


def ensure_directory(path: Path) -> None:
    """Ensure directory exists."""
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created directory: {path}")


def run_command(command: str) -> str:
    """Run a command and return its output."""
    result = subprocess.run(
        command, shell=True, check=True, text=True, capture_output=True
    )
    return result.stdout


def generate_base_requirements() -> None:
    """Generate base requirements.txt from Poetry."""
    print("Generating base requirements.txt...")
    output = run_command("poetry export --without-hashes --format=requirements.txt")
    
    with open(Path("requirements.txt"), "w") as f:
        f.write("# Base requirements generated from Poetry\n")
        f.write("# Generated on: " + run_command("date").strip() + "\n\n")
        f.write(output)
    
    print("✓ Generated requirements.txt")


def generate_group_requirements(group: str, output_file: str) -> None:
    """Generate group-specific requirements.txt from Poetry."""
    print(f"Generating {output_file}...")
    
    command = f"poetry export --without-hashes --with={group} --format=requirements.txt"
    output = run_command(command)
    
    with open(Path(output_file), "w") as f:
        f.write(f"# {group.capitalize()} requirements generated from Poetry\n")
        f.write("# Generated on: " + run_command("date").strip() + "\n\n")
        f.write(output)
    
    print(f"✓ Generated {output_file}")


def generate_streamlit_requirements() -> None:
    """Generate Streamlit Cloud compatible requirements."""
    print("Generating Streamlit Cloud requirements.txt...")
    
    # For Streamlit Cloud we need main dependencies plus personality dependencies
    command = "poetry export --without-hashes --with=personality --format=requirements.txt"
    output = run_command(command)
    
    with open(Path("requirements/requirements-streamlit.txt"), "w") as f:
        f.write("# Streamlit Cloud deployment requirements\n")
        f.write("# Generated on: " + run_command("date").strip() + "\n\n")
        f.write(output)
    
    print("✓ Generated requirements/requirements-streamlit.txt")


def main() -> None:
    """Main entry point."""
    # Ensure we're in the project root
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir.parent)
    
    # Ensure requirements directory exists
    ensure_directory(Path("requirements"))
    
    # Generate base requirements.txt in the root directory
    generate_base_requirements()
    
    # Generate environment-specific requirements
    generate_group_requirements("dev", "requirements/requirements-dev.txt")
    generate_group_requirements("monitoring", "requirements/requirements-monitoring.txt")
    generate_group_requirements("deployment", "requirements/requirements-deployment.txt")
    
    # Generate Streamlit Cloud specific requirements
    generate_streamlit_requirements()
    
    print("\nAll requirements files generated successfully!")
    print("To install development dependencies: pip install -r requirements/requirements-dev.txt")


if __name__ == "__main__":
    main() 