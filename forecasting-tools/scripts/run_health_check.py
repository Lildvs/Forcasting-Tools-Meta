#!/usr/bin/env python
"""
Personality System Health Check Runner

This script provides a convenient way to run the personality system
health check and generate a report of the system's status.

Usage:
    python -m forecasting_tools.scripts.run_health_check [--level LEVEL] [--output FILE]

Options:
    --level LEVEL    Check level: minimal, standard, or comprehensive (default: standard)
    --output FILE    Path to save the report (default: personality_health_report.json)
    --verbose        Enable verbose logging
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

from forecasting_tools.personality_management.health_check import check_system_health

def main():
    """Run the personality system health check."""
    parser = argparse.ArgumentParser(description="Run health checks on the personality system")
    parser.add_argument(
        "--level", 
        choices=["minimal", "standard", "comprehensive"],
        default="standard",
        help="Level of health check to perform"
    )
    parser.add_argument(
        "--output",
        default=f"personality_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to save the health check report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print(f"Running {args.level} health check on personality system...")
    
    results = check_system_health(args.level, args.output)
    
    # Print summary
    print(f"\nHealth Check Summary: {results['overall_status'].upper()}")
    print(f"Components checked: {len(results['components'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"- {warning}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"- {error}")
    
    print(f"\nDetailed report saved to: {args.output}")
        
    # Exit with appropriate status code
    if results['overall_status'] == "error":
        sys.exit(1)

if __name__ == "__main__":
    main() 