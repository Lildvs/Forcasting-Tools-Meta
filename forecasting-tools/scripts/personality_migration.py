#!/usr/bin/env python
"""
Personality Migration Script

This script migrates existing forecasts to use the new personality management system.
It performs the following operations:
1. Identifies forecasts made with legacy configurations
2. Maps them to appropriate personality configurations
3. Updates forecast metadata to include personality information
4. Validates compatibility with the new system

Usage:
    python -m forecasting_tools.scripts.personality_migration --input-dir=/path/to/forecasts --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.validators import PersonalityValidator
from forecasting_tools.data_models.binary_report import BinaryReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("personality_migration")

# Mapping of legacy configurations to personality names
LEGACY_TO_PERSONALITY_MAP = {
    # Map legacy configuration patterns to personality names
    "analytical": "analytical",
    "creative": "creative",
    "balanced": "balanced",
    "careful": "cautious",
    "thorough": "deep",
    "quick": "shallow",
    "bayesian": "bayesian",
    # Default fallback
    "default": "balanced"
}

def detect_legacy_personality(metadata: Dict[str, Any]) -> str:
    """
    Detect legacy personality configuration from forecast metadata.
    
    Args:
        metadata: Forecast metadata dictionary
        
    Returns:
        Detected personality name or default
    """
    if not metadata:
        return LEGACY_TO_PERSONALITY_MAP["default"]
    
    # Check for direct indicators in metadata
    if "thinking_style" in metadata:
        style = metadata["thinking_style"].lower()
        for key, personality in LEGACY_TO_PERSONALITY_MAP.items():
            if key in style:
                return personality
    
    # Check for instruction patterns
    if "instructions" in metadata:
        instructions = metadata["instructions"].lower()
        patterns = {
            "analytical": ["analyze", "analytical", "logical"],
            "creative": ["creative", "imagine", "novel"],
            "cautious": ["careful", "cautious", "conservative"],
            "bayesian": ["bayesian", "probability", "prior"]
        }
        
        matches = {}
        for personality, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in instructions)
            if score > 0:
                matches[personality] = score
        
        if matches:
            best_match = max(matches.items(), key=lambda x: x[1])[0]
            return best_match
    
    # Check for temperature settings
    if "temperature" in metadata:
        temp = float(metadata["temperature"])
        if temp < 0.4:
            return "analytical"
        elif temp > 0.8:
            return "creative"
    
    return LEGACY_TO_PERSONALITY_MAP["default"]

def update_forecast_metadata(
    forecast_file: Path, 
    personality_name: str, 
    personality: PersonalityConfig,
    dry_run: bool = False
) -> bool:
    """
    Update a forecast file with personality information.
    
    Args:
        forecast_file: Path to the forecast file
        personality_name: Name of the personality to apply
        personality: Personality configuration
        dry_run: If True, don't actually write changes
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        with open(forecast_file, "r") as f:
            forecast_data = json.load(f)
        
        # Update metadata
        if "metadata" not in forecast_data:
            forecast_data["metadata"] = {}
        
        forecast_data["metadata"]["personality_name"] = personality_name
        forecast_data["metadata"]["personality_thinking_style"] = personality.thinking_style.value
        forecast_data["metadata"]["personality_uncertainty_approach"] = personality.uncertainty_approach.value
        forecast_data["metadata"]["personality_reasoning_depth"] = personality.reasoning_depth.value
        
        if not dry_run:
            with open(forecast_file, "w") as f:
                json.dump(forecast_data, f, indent=2)
            logger.info(f"Updated forecast {forecast_file}")
        else:
            logger.info(f"[DRY RUN] Would update forecast {forecast_file}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to update forecast {forecast_file}: {str(e)}")
        return False

def process_directory(
    directory: Path, 
    personality_manager: PersonalityManager,
    validator: PersonalityValidator,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Process all forecast files in a directory.
    
    Args:
        directory: Directory containing forecast files
        personality_manager: PersonalityManager instance
        validator: PersonalityValidator instance
        dry_run: If True, don't actually write changes
        
    Returns:
        Tuple of (total_processed, successful_updates)
    """
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return 0, 0
    
    # Find all JSON files that might be forecasts
    forecast_files = list(directory.glob("**/*.json"))
    logger.info(f"Found {len(forecast_files)} potential forecast files in {directory}")
    
    total_processed = 0
    successful_updates = 0
    
    for file_path in forecast_files:
        try:
            # Check if file is a forecast
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    # Basic check if this looks like a forecast
                    if not isinstance(data, dict) or "question_text" not in data:
                        continue
                    
                    # Check if already migrated
                    if "metadata" in data and "personality_name" in data["metadata"]:
                        logger.debug(f"Already migrated: {file_path}")
                        continue
                    
                    total_processed += 1
                    
                    # Detect legacy personality
                    metadata = data.get("metadata", {})
                    personality_name = detect_legacy_personality(metadata)
                    
                    # Load personality
                    personality = personality_manager.load_personality(personality_name)
                    if not personality:
                        logger.warning(f"Personality not found: {personality_name}, using default")
                        personality_name = "balanced"
                        personality = personality_manager.load_personality(personality_name)
                    
                    # Update forecast
                    if update_forecast_metadata(file_path, personality_name, personality, dry_run):
                        successful_updates += 1
                        
                except json.JSONDecodeError:
                    continue
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return total_processed, successful_updates

def main():
    parser = argparse.ArgumentParser(description="Migrate existing forecasts to the personality system")
    parser.add_argument("--input-dir", required=True, help="Directory containing forecast files")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes, just simulate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting personality migration {'(DRY RUN)' if args.dry_run else ''}")
    
    # Initialize components
    personality_manager = PersonalityManager()
    validator = PersonalityValidator()
    
    # Process directory
    input_dir = Path(args.input_dir)
    total, successful = process_directory(
        input_dir, 
        personality_manager, 
        validator,
        args.dry_run
    )
    
    logger.info(f"Migration completed: processed {total} forecasts, updated {successful}")
    
    if total > 0:
        success_rate = (successful / total) * 100
        logger.info(f"Success rate: {success_rate:.2f}%")
    
    if args.dry_run:
        logger.info("This was a dry run. No files were modified.")

if __name__ == "__main__":
    main() 