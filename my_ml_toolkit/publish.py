#!/usr/bin/env python3
"""
Build and publish script for my_ml_toolkit
"""
import subprocess
import sys

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Success: {description}")

def main():
    print("🚀 My ML Toolkit - PyPI Publishing Guide\n")
    
    # Step 1: Check environment
    run_command("python --version", "Check Python version")
    
    # Step 2: Install build tools
    run_command(
        "pip install --upgrade pip setuptools wheel twine",
        "Install/upgrade build tools"
    )
    
    # Step 3: Clean previous builds
    run_command(
        "rm -rf build/ dist/ *.egg-info/",
        "Clean previous builds"
    )
    
    # Step 4: Build distribution
    run_command(
        "python -m build",
        "Build source distribution and wheel"
    )
    
    # Step 5: Check distribution
    run_command(
        "twine check dist/*",
        "Check distribution validity"
    )
    
    # Step 6: Test upload (optional)
    print(f"\n{'='*70}")
    print("📝 Next steps:")
    print(f"{'='*70}")
    print("1. Test upload (recommended first time):")
    print("   twine upload --repository testpypi dist/*")
    print("\n2. Production upload:")
    print("   twine upload dist/*")
    print("\nNote: You'll need PyPI API token (get from https://pypi.org/manage/account/)")
    print("      Create ~/.pypirc with your credentials\n")

if __name__ == "__main__":
    main()
