"""
Main script to run all tasks sequentially

This script runs all four tasks in order:
1. Task 1: EDA and Statistical Analysis
2. Task 2: DVC Setup
3. Task 3: Hypothesis Testing
4. Task 4: Machine Learning Modeling
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from task1_eda.eda_analysis import EDAAnalyzer
from task2_dvc.dvc_setup import setup_dvc
from task3_hypothesis.hypothesis_tests import HypothesisTester
from task4_modeling.train_models import InsuranceModelTrainer


def main():
    """Run all tasks"""
    data_path = "data/insurance_data.csv"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please add your insurance data file to the data/ directory")
        return
    
    print("="*70)
    print("INSURANCE RISK ANALYTICS - COMPLETE PIPELINE")
    print("="*70)
    
    # Task 1: EDA
    print("\n" + "="*70)
    print("TASK 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    analyzer = EDAAnalyzer(data_path)
    analyzer.run_full_eda()
    
    # Task 2: DVC Setup (informational)
    print("\n" + "="*70)
    print("TASK 2: DATA VERSION CONTROL")
    print("="*70)
    print("Note: DVC setup should be run manually using:")
    print("  python src/task2_dvc/dvc_setup.py")
    print("  dvc add data/insurance_data.csv")
    
    # Task 3: Hypothesis Testing
    print("\n" + "="*70)
    print("TASK 3: HYPOTHESIS TESTING")
    print("="*70)
    tester = HypothesisTester(data_path)
    tester.run_all_tests(alpha=0.05)
    
    # Task 4: Modeling
    print("\n" + "="*70)
    print("TASK 4: MACHINE LEARNING MODELING")
    print("="*70)
    trainer = InsuranceModelTrainer(data_path)
    trainer.run_all_modeling()
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED!")
    print("="*70)
    print("\nCheck the reports/ directory for all generated reports and visualizations.")


if __name__ == "__main__":
    main()

