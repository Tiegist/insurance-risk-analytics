"""
Data Version Control (DVC) Setup Script

This script helps set up DVC for the insurance risk analytics project.
Run this script to initialize DVC and configure local remote storage.
"""

import subprocess
import sys
from pathlib import Path
import os


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description} failed")
        print(f"  Error message: {e.stderr}")
        return False


def setup_dvc():
    """Set up DVC for the project"""
    print("="*50)
    print("DVC SETUP")
    print("="*50)
    
    # Check if DVC is installed
    print("\n1. Checking DVC installation...")
    try:
        subprocess.run(["dvc", "--version"], check=True, capture_output=True)
        print("✓ DVC is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ DVC is not installed. Installing...")
        run_command("pip install dvc", "Installing DVC")
    
    # Initialize DVC
    if not Path(".dvc").exists():
        run_command("dvc init", "Initializing DVC")
    else:
        print("✓ DVC already initialized")
    
    # Create local storage directory
    storage_path = Path("dvc_storage")
    if not storage_path.exists():
        storage_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created storage directory: {storage_path.absolute()}")
    else:
        print(f"✓ Storage directory already exists: {storage_path.absolute()}")
    
    # Add local remote storage
    print("\n2. Configuring local remote storage...")
    run_command(f'dvc remote add -d localstorage "{storage_path.absolute()}"', 
                "Adding local remote storage")
    
    # Check if data directory exists
    data_path = Path("data")
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created data directory: {data_path.absolute()}")
    
    print("\n" + "="*50)
    print("DVC SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Add your data file to the data/ directory")
    print("2. Run: dvc add data/your_data_file.csv")
    print("3. Commit the .dvc file: git add data/your_data_file.csv.dvc")
    print("4. Push data to storage: dvc push")


def add_data_file(data_file_path: str):
    """Add a data file to DVC tracking"""
    data_path = Path(data_file_path)
    
    if not data_path.exists():
        print(f"✗ Error: Data file not found at {data_path}")
        return False
    
    print(f"\nAdding {data_path} to DVC...")
    return run_command(f"dvc add {data_path}", f"Adding {data_path.name} to DVC")


if __name__ == "__main__":
    setup_dvc()
    
    # If a data file is provided as argument, add it
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        add_data_file(data_file)

