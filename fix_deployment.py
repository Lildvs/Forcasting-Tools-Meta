#!/usr/bin/env python3
"""
Comprehensive Fix for Deployment Issues

This script addresses all identified issues in the deployed application:
1. Personality path issues
2. Template loading errors
3. BinaryForecast attribute error
"""

import os
import sys
import json
import shutil
from pathlib import Path

def fix_deployment_issues():
    """
    Apply fixes for the deployed environment.
    """
    print("Starting deployment fixes...")
    
    # Determine if we're in deployment or local environment
    in_deployment = os.path.exists("/mount/src/forcasting-tools-meta")
    
    if in_deployment:
        base_dir = Path("/mount/src/forcasting-tools-meta/forecasting-tools")
    else:
        current_dir = Path(os.getcwd())
        base_dir = current_dir / "forecasting-tools"
    
    if not base_dir.exists():
        print(f"Error: Base directory not found at {base_dir}")
        return False
    
    print(f"Base directory: {base_dir}")
    
    # Define important paths
    personalities_dir = base_dir / "forecasting_tools" / "personality_management" / "personalities"
    templates_dir = base_dir / "forecasting_tools" / "personality_management" / "templates"
    
    # Fix 1: Copy analytical.yaml to a location that's definitely accessible
    if not fix_personalities(personalities_dir):
        return False
    
    # Fix 2: Ensure the binary_forecast_prompt template exists and is valid
    if not fix_templates(templates_dir):
        return False
    
    # Fix 3: Add a compatibility layer for BinaryForecast objects
    if not fix_binary_forecast_class(base_dir):
        return False
    
    print("\nAll fixes applied successfully!")
    return True

def fix_personalities(personalities_dir):
    """
    Fix personality file issues.
    """
    print(f"\n1. Fixing personality files in {personalities_dir}")
    
    if not personalities_dir.exists():
        try:
            print(f"  Creating personalities directory: {personalities_dir}")
            personalities_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  Error creating personalities directory: {e}")
            return False
    
    # Create default analytical.yaml if it doesn't exist
    analytical_file = personalities_dir / "analytical.yaml"
    
    if not analytical_file.exists():
        print(f"  Creating analytical.yaml: {analytical_file}")
        analytical_content = """# Analytical Personality Configuration
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
"""
        try:
            with open(analytical_file, "w", encoding="utf-8") as f:
                f.write(analytical_content)
            print("  Successfully created analytical.yaml")
        except Exception as e:
            print(f"  Error creating analytical.yaml: {e}")
            return False
    else:
        print(f"  analytical.yaml already exists")
    
    # Make sure all personality files have correct permissions
    try:
        for yaml_file in personalities_dir.glob("*.yaml"):
            os.chmod(yaml_file, 0o644)  # rw-r--r--
            print(f"  Fixed permissions for {yaml_file.name}")
    except Exception as e:
        print(f"  Error fixing permissions: {e}")
        # Non-fatal, continue
    
    return True

def fix_templates(templates_dir):
    """
    Fix template loading issues.
    """
    print(f"\n2. Fixing templates in {templates_dir}")
    
    if not templates_dir.exists():
        try:
            print(f"  Creating templates directory: {templates_dir}")
            templates_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  Error creating templates directory: {e}")
            return False
    
    # Create the binary_forecast_prompt template if it doesn't exist or is invalid
    template_file = templates_dir / "binary_forecast_prompt.json"
    
    # Check if file exists and is valid JSON
    needs_fix = True
    if template_file.exists():
        try:
            with open(template_file, "r", encoding="utf-8") as f:
                json.load(f)  # Test if it's valid JSON
            print(f"  Template file exists and is valid: {template_file}")
            needs_fix = False
        except json.JSONDecodeError:
            print(f"  Template file exists but is invalid JSON: {template_file}")
            needs_fix = True
        except Exception as e:
            print(f"  Error reading template file: {e}")
            needs_fix = True
    
    if needs_fix:
        print(f"  Creating/fixing template file: {template_file}")
        template_content = """{
    "template": "You are a {{thinking_style}} forecaster with a {{uncertainty_approach}} approach to uncertainty. You have been asked to forecast the probability of a binary event.\n\nQuestion: {{question}}\n\n{{#if resolution_criteria}}Resolution Criteria: {{resolution_criteria}}{{/if}}\n\n{{#if background_info}}Background Information: {{background_info}}{{/if}}\n\n{{#if research}}Research:\n{{research}}{{/if}}\n\nIn responding to this forecasting request:\n- {{reasoning_prefix}}\n- {{calibration_guidance}}\n- {{uncertainty_handling}}\n\nProvide your reasoning, then your final probability estimate as a percentage.",
    "metadata": {
        "description": "Template for generating binary forecasts",
        "tags": ["binary", "forecast", "probability"]
    }
}"""
        try:
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template_content)
            print("  Successfully created/fixed binary_forecast_prompt.json")
        except Exception as e:
            print(f"  Error creating/fixing template file: {e}")
            return False
    
    return True

def fix_binary_forecast_class(base_dir):
    """
    Fix BinaryForecast class attribute error.
    """
    print("\n3. Adding compatibility layer for BinaryForecast class")
    
    # Create a compatibility.py file
    compat_dir = base_dir / "forecasting_tools" / "compat"
    compat_file = compat_dir / "compatibility.py"
    
    if not compat_dir.exists():
        try:
            compat_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  Error creating compat directory: {e}")
            return False
    
    compat_content = """\"\"\"
Compatibility layer for handling API changes between versions.
\"\"\"

class BinaryForecastCompat:
    \"\"\"
    Monkey patch to add backward compatibility to BinaryForecast class.
    \"\"\"
    
    @classmethod
    def apply_patches(cls):
        \"\"\"Apply all compatibility patches.\"\"\"
        cls.patch_binary_forecast()
    
    @classmethod
    def patch_binary_forecast(cls):
        \"\"\"
        Add compatibility for BinaryForecast objects.
        Adds prediction property that returns probability.
        \"\"\"
        try:
            from forecasting_tools.data_models.binary_report import BinaryForecast
            
            # Only add the property if it doesn't already exist
            if not hasattr(BinaryForecast, "prediction"):
                # Add prediction property that returns probability
                BinaryForecast.prediction = property(lambda self: self.probability)
                print("  Successfully added prediction property to BinaryForecast")
            else:
                print("  BinaryForecast already has prediction property")
                
        except ImportError as e:
            print(f"  Error importing BinaryForecast: {e}")
            return False
        except Exception as e:
            print(f"  Error patching BinaryForecast: {e}")
            return False
        
        return True

# Create empty __init__.py to make it a proper package
init_file = __file__.replace("compatibility.py", "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("# Compatibility layer for handling API changes between versions.")
"""
    
    try:
        with open(compat_file, "w", encoding="utf-8") as f:
            f.write(compat_content)
        print(f"  Created compatibility layer: {compat_file}")
        
        # Create __init__.py
        init_file = compat_dir / "__init__.py"
        with open(init_file, "w", encoding="utf-8") as f:
            f.write("# Compatibility layer for handling API changes between versions.")
        print(f"  Created __init__.py: {init_file}")
    except Exception as e:
        print(f"  Error creating compatibility layer: {e}")
        return False
    
    # Create startup hook to apply patches
    startup_dir = base_dir / "forecasting_tools"
    startup_file = startup_dir / "startup.py"
    
    startup_content = """\"\"\"
Startup module to apply patches and fixes on application startup.
\"\"\"

def apply_startup_fixes():
    \"\"\"Apply all necessary fixes and patches on startup.\"\"\"
    try:
        # Apply compatibility patches
        from forecasting_tools.compat.compatibility import BinaryForecastCompat
        BinaryForecastCompat.apply_patches()
        
        # Fix any other startup issues here
        
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error applying startup fixes: {e}")
        return False

# Run startup fixes
apply_startup_fixes()
"""
    
    try:
        with open(startup_file, "w", encoding="utf-8") as f:
            f.write(startup_content)
        print(f"  Created startup hook: {startup_file}")
        
        # Ensure startup is imported
        app_file = base_dir / "streamlit_app.py"
        if app_file.exists():
            with open(app_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Check if startup is already imported
            if "from forecasting_tools.startup import" not in content:
                # Find first import
                import_lines = content.split("\n")
                for i, line in enumerate(import_lines):
                    if line.startswith("import ") or line.startswith("from "):
                        # Insert our import after the first import
                        import_lines.insert(i+1, "# Apply fixes on startup")
                        import_lines.insert(i+2, "from forecasting_tools.startup import apply_startup_fixes")
                        break
                
                # Write the updated content
                with open(app_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(import_lines))
                print(f"  Updated {app_file} to import startup module")
    except Exception as e:
        print(f"  Error creating startup hook: {e}")
        # Non-fatal, continue
    
    return True

if __name__ == "__main__":
    success = fix_deployment_issues()
    sys.exit(0 if success else 1) 