#!/usr/bin/env python3
"""
Direct Fix Script

This script directly modifies the streamlit_app.py file to add inline fixes
rather than relying on a separate module that might not be imported early enough.
"""

import os
import re
import sys
from pathlib import Path

def apply_direct_fix():
    """Apply fixes directly to the streamlit_app.py file"""
    print("Applying direct fixes...")
    
    # Find streamlit_app.py
    current_dir = Path(os.getcwd())
    streamlit_app_path = current_dir / "forecasting-tools" / "streamlit_app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: streamlit_app.py not found at {streamlit_app_path}")
        return False
    
    # Read streamlit_app.py
    try:
        with open(streamlit_app_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create backup
        with open(streamlit_app_path.with_suffix(".py.bak"), "w", encoding="utf-8") as f:
            f.write(content)
        
        # Add inline fix at the very beginning of the file (after existing imports)
        fix_code = """
# ===== DIRECT FIX FOR ATTRIBUTE ERROR =====
# This code adds the 'prediction' property to BinaryForecast class
try:
    from forecasting_tools.data_models.binary_report import BinaryForecast
    if not hasattr(BinaryForecast, "prediction"):
        print("Adding prediction property to BinaryForecast class...")
        BinaryForecast.prediction = property(lambda self: self.probability)
except Exception as e:
    import logging
    logging.getLogger(__name__).error(f"Error applying BinaryForecast fix: {e}")

# Create personalities directory and template directory if they don't exist
try:
    import os
    from pathlib import Path
    base_dir = Path(__file__).parent
    
    # Create personalities directory
    personalities_dir = base_dir / "forecasting_tools" / "personality_management" / "personalities"
    if not personalities_dir.exists():
        print(f"Creating personalities directory: {personalities_dir}")
        personalities_dir.mkdir(exist_ok=True, parents=True)
    
    # Create analytical.yaml if it doesn't exist
    analytical_file = personalities_dir / "analytical.yaml"
    if not analytical_file.exists():
        print(f"Creating analytical.yaml: {analytical_file}")
        analytical_content = '''# Analytical Personality Configuration
# A personality that emphasizes deep analysis, systematic thinking, and data-driven reasoning

name: analytical
description: A methodical forecaster who relies heavily on systematic analysis, data-driven reasoning, and structured frameworks

# Core personality traits
reasoning_depth: deep
uncertainty_approach: explicit
thinking_style: analytical
temperature: 0.3

# Template variables for prompt customization
template_variables:
  reasoning_prefix: "I will conduct a thorough, structured analysis of the evidence and systematically evaluate different hypotheses."
  calibration_guidance: "I should quantify uncertainties explicitly and ensure my confidence levels reflect the available evidence."
  uncertainty_handling: "I will break down complex uncertainties into component parts and address each systematically."

# Custom traits
traits:
  # How much the forecaster relies on data vs. intuition
  data_reliance:
    description: How much the forecaster relies on data versus intuition
    value: 0.9  # 0.0 = pure intuition, 1.0 = pure data
'''
        with open(analytical_file, "w", encoding="utf-8") as f:
            f.write(analytical_content)
    
    # Create templates directory
    templates_dir = base_dir / "forecasting_tools" / "personality_management" / "templates"
    if not templates_dir.exists():
        print(f"Creating templates directory: {templates_dir}")
        templates_dir.mkdir(exist_ok=True, parents=True)
    
    # Create binary_forecast_prompt.json
    template_file = templates_dir / "binary_forecast_prompt.json"
    if not template_file.exists():
        print(f"Creating binary_forecast_prompt.json: {template_file}")
        template_content = '''{
    "template": "You are a {{thinking_style}} forecaster with a {{uncertainty_approach}} approach to uncertainty. You have been asked to forecast the probability of a binary event.\\n\\nQuestion: {{question}}\\n\\n{{#if resolution_criteria}}Resolution Criteria: {{resolution_criteria}}{{/if}}\\n\\n{{#if background_info}}Background Information: {{background_info}}{{/if}}\\n\\n{{#if research}}Research:\\n{{research}}{{/if}}\\n\\nIn responding to this forecasting request:\\n- {{reasoning_prefix}}\\n- {{calibration_guidance}}\\n- {{uncertainty_handling}}\\n\\nProvide your reasoning, then your final probability estimate as a percentage.",
    "metadata": {
        "description": "Template for generating binary forecasts",
        "tags": ["binary", "forecast", "probability"]
    }
}'''
        with open(template_file, "w", encoding="utf-8") as f:
            f.write(template_content)

except Exception as e:
    import logging
    logging.getLogger(__name__).error(f"Error setting up directories: {e}")
# ===== END OF DIRECT FIX =====
"""
        
        # Find first import statement in the file
        pattern = r"^(import|from)\s+"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            # Find all imports as a block
            pos = match.start()
            
            # Search for the end of imports
            lines = content.splitlines()
            end_pos = 0
            in_imports = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if re.match(pattern, line):
                    in_imports = True
                    end_pos = i
                elif in_imports and line and not line.startswith("#") and not re.match(pattern, line):
                    # This line is not an import and not a comment - end of import block
                    break
            
            # Insert our fix code after the imports
            new_content = "\n".join(lines[:end_pos+1]) + "\n" + fix_code + "\n" + "\n".join(lines[end_pos+1:])
            
            # Write the modified content
            with open(streamlit_app_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            print(f"Successfully applied direct fix to {streamlit_app_path}")
            return True
        else:
            print("Error: Could not find import statements in streamlit_app.py")
            return False
            
    except Exception as e:
        print(f"Error applying direct fix: {e}")
        return False

if __name__ == "__main__":
    success = apply_direct_fix()
    sys.exit(0 if success else 1) 