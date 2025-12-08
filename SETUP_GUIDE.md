# Setup Guide

This guide will help you set up and run the Insurance Risk Analytics project.

## Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package manager)

## Step 1: Clone/Setup Repository

If you haven't already, initialize the repository:
```bash
git init
```

## Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Add Your Data

1. Download the insurance data from the provided link
2. Place the CSV file in the `data/` directory
3. Rename it to `insurance_data.csv` (or update the path in the scripts)

## Step 5: Initialize DVC (Task 2)

```bash
# Run the DVC setup script
python src/task2_dvc/dvc_setup.py

# Add your data file to DVC
dvc add data/insurance_data.csv

# Commit the .dvc file
git add data/insurance_data.csv.dvc data/.gitignore
git commit -m "Add data file to DVC"
```

## Step 6: Run Individual Tasks

### Task 1: EDA
```bash
python src/task1_eda/eda_analysis.py
```

### Task 2: DVC Setup
```bash
python src/task2_dvc/dvc_setup.py
dvc add data/insurance_data.csv
```

### Task 3: Hypothesis Testing
```bash
python src/task3_hypothesis/hypothesis_tests.py
```

### Task 4: Machine Learning Modeling
```bash
python src/task4_modeling/train_models.py
```

### Run All Tasks
Run each task individually as shown above.

## Step 7: Git Workflow

### Create Task Branches

```bash
# Task 1
git checkout -b task-1
# ... make changes ...
git add .
git commit -m "feat: complete task 1 EDA analysis"
git push origin task-1

# Task 2
git checkout main
git checkout -b task-2
# ... make changes ...
git add .
git commit -m "feat: set up DVC for data versioning"
git push origin task-2

# Task 3
git checkout main
git checkout -b task-3
# ... make changes ...
git add .
git commit -m "feat: implement hypothesis testing"
git push origin task-3

# Task 4
git checkout main
git checkout -b task-4
# ... make changes ...
git add .
git commit -m "feat: build ML models for risk prediction"
git push origin task-4
```

### Create Pull Requests

After completing each task, create a Pull Request to merge into main:
1. Push your branch to GitHub
2. Create a Pull Request on GitHub
3. Review and merge

## Step 8: View Results

All reports and visualizations are saved in the `reports/` directory:
- `reports/task1_eda/` - EDA reports and visualizations
- `reports/task3_hypothesis/` - Hypothesis test results
- `reports/task4_modeling/` - Model evaluation and feature importance

## Troubleshooting

### Data File Not Found
- Ensure your data file is in the `data/` directory
- Check the file name matches what's expected in the scripts
- Update the `data_path` variable in the scripts if needed

### DVC Issues
- Make sure DVC is installed: `pip install dvc`
- Initialize DVC: `dvc init`
- Check DVC remote: `dvc remote list`

### Import Errors
- Make sure you're in the project root directory
- Activate your virtual environment
- Install all dependencies: `pip install -r requirements.txt`

## Next Steps

1. Review the generated reports in the `reports/` directory
2. Customize the analysis based on your findings
3. Create your final report (Medium blog post format)
4. Submit your GitHub repository link

## Support

For questions or issues, refer to:
- Project README.md
- 10 Academy documentation
- GitHub Issues (if using GitHub)

