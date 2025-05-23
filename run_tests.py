#!/usr/bin/env python
"""
Test Runner Script

This script provides a convenient way to run tests for the forecasting tools project
with various configurations and report options.
"""

import os
import sys
import argparse
import subprocess
import datetime
import shutil
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for forecasting tools")
    
    # Test selection options
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--module", type=str, help="Run tests for a specific module")
    parser.add_argument("--file", type=str, help="Run a specific test file")
    parser.add_argument("--test", type=str, help="Run a specific test case or method")
    
    # Coverage options
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="count", default=0, 
                      help="Increase verbosity (can be used multiple times)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--output-dir", type=str, default="test-results",
                      help="Directory for test reports")
    
    # Other options
    parser.add_argument("--cleanup", action="store_true", 
                      help="Remove temporary files after tests")
    parser.add_argument("--skip-db", action="store_true",
                      help="Skip tests that require database setup")
    parser.add_argument("--junit-xml", action="store_true",
                      help="Generate JUnit XML report")
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark tests")
    
    return parser.parse_args()


def build_pytest_command(args):
    """Build the pytest command based on arguments."""
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose > 0:
        cmd.extend(["-" + "v" * args.verbose])
    if args.quiet:
        cmd.append("-q")
    
    # Add test selection
    if args.unit:
        cmd.append("tests/unit/")
    elif args.integration:
        cmd.append("tests/integration/")
    elif args.all:
        cmd.append("tests/")
    elif args.file:
        cmd.append(args.file)
    elif args.module:
        cmd.append(f"tests/unit/test_{args.module}.py")
        cmd.append(f"tests/integration/test_{args.module}.py")
    
    # Add specific test
    if args.test:
        cmd.append(f"-k {args.test}")
    
    # Add coverage
    if args.coverage:
        cmd.append("--cov=forecasting_tools")
        if args.html:
            cmd.append("--cov-report=html")
        if args.xml:
            cmd.append("--cov-report=xml")
        if not (args.html or args.xml):
            cmd.append("--cov-report=term")
    
    # Add JUnit XML report
    if args.junit_xml:
        os.makedirs(args.output_dir, exist_ok=True)
        cmd.append(f"--junitxml={args.output_dir}/junit.xml")
    
    # Skip database tests if requested
    if args.skip_db:
        cmd.append("-k not database")
    
    # Add benchmark tests if requested
    if args.benchmark:
        cmd.append("--benchmark-enable")
        cmd.append(f"--benchmark-json={args.output_dir}/benchmark.json")
    
    return cmd


def run_tests(args):
    """Run tests with the specified configuration."""
    # Default to all tests if nothing specific is selected
    if not any([args.unit, args.integration, args.all, args.file, args.module, args.test]):
        args.all = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build and run pytest command
    cmd = build_pytest_command(args)
    cmd_str = " ".join(cmd)
    
    print(f"Running tests with command: {cmd_str}")
    start_time = datetime.datetime.now()
    
    try:
        result = subprocess.run(cmd, capture_output=not args.verbose, text=True, check=False)
        
        # Store the output
        with open(f"{args.output_dir}/pytest_output.txt", "w") as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
        
        # Report result
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nTests completed in {duration:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("All tests passed successfully!")
        else:
            print("Some tests failed. See the output for details.")
            if not args.verbose and result.stdout:
                print("\nTest output:")
                print(result.stdout)
        
        return result.returncode
    
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return 1


def cleanup_temp_files():
    """Clean up temporary files created during tests."""
    # Remove SQLite database files
    for db_file in Path(".").glob("*.db"):
        try:
            db_file.unlink()
            print(f"Removed temporary database file: {db_file}")
        except Exception as e:
            print(f"Failed to remove {db_file}: {str(e)}")
    
    # Remove __pycache__ directories
    for pycache in Path(".").glob("**/__pycache__"):
        if pycache.is_dir():
            try:
                shutil.rmtree(pycache)
                print(f"Removed {pycache}")
            except Exception as e:
                print(f"Failed to remove {pycache}: {str(e)}")
    
    # Remove .pytest_cache
    pytest_cache = Path(".pytest_cache")
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache)
            print("Removed .pytest_cache")
        except Exception as e:
            print(f"Failed to remove .pytest_cache: {str(e)}")


def main():
    """Main entry point."""
    # Set up environment variables
    os.environ["PYTHONPATH"] = os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")])
    os.environ["FORECASTING_TOOLS_ENV"] = "test"
    
    # Parse arguments and run tests
    args = parse_args()
    exit_code = run_tests(args)
    
    # Clean up if requested
    if args.cleanup:
        cleanup_temp_files()
    
    # Print report locations
    if args.coverage and args.html:
        print(f"\nCoverage report: file://{os.getcwd()}/htmlcov/index.html")
    if args.junit_xml:
        print(f"\nJUnit XML report: {args.output_dir}/junit.xml")
    if args.benchmark:
        print(f"\nBenchmark report: {args.output_dir}/benchmark.json")
    
    # Exit with the same code as pytest
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 