#!/usr/bin/env python3
"""
Test runner for the new model management system.

This script runs the comprehensive test suite for the new model managers
and provides detailed output about test results.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Run the test suite for the new model management system."""
    
    print("MASX AI Model Manager Test Suite")
    print("=" * 50)
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Test files to run
    test_files = [
        "tests/test_model_managers.py",
        "tests/test_mock_models.py",
    ]
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        *test_files
    ]
    
    print(f"Running tests: {' '.join(test_files)}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n" + "=" * 50)
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False


def run_specific_test(test_name: str):
    """Run a specific test by name."""
    
    print(f"Running specific test: {test_name}")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--color=yes",
        "-k", test_name
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_tests()
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
