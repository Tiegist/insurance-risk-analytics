# Project Structure

This document describes the structure of the Insurance Risk Analytics project.

## Directory Structure

```
insurance-risk-analytics/
│
├── data/                          # Data files (tracked with DVC)
│   └── .gitkeep
│
├── notebooks/                     # Jupyter notebooks for exploration
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── task1_eda/                 # Task 1: EDA and Statistical Analysis
│   │   ├── __init__.py
│   │   └── eda_analysis.py        # Main EDA script
│   │
│   ├── task2_dvc/                 # Task 2: Data Version Control
│   │   ├── __init__.py
│   │   └── dvc_setup.py           # DVC setup script
│   │
│   ├── task3_hypothesis/          # Task 3: A/B Hypothesis Testing
│   │   ├── __init__.py
│   │   └── hypothesis_tests.py   # Hypothesis testing script
│   │
│   └── task4_modeling/            # Task 4: Machine Learning Modeling
│       ├── __init__.py
│       └── train_models.py        # ML model training script
│
├── reports/                        # Generated reports and visualizations
│   ├── task1_eda/                 # EDA reports
│   ├── task3_hypothesis/          # Hypothesis test results
│   └── task4_modeling/            # Model evaluation results
│
├── models/                         # Saved trained models
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   └── test_eda.py
│
├── .github/                        # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline
│
├── README.md                       # Main project documentation
├── SETUP_GUIDE.md                 # Setup instructions
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── run_all_tasks.py               # Main script to run all tasks
```

## Key Files

### Main Scripts

- **`run_all_tasks.py`**: Runs all four tasks sequentially
- **`src/task1_eda/eda_analysis.py`**: Performs comprehensive EDA
- **`src/task2_dvc/dvc_setup.py`**: Sets up DVC for data versioning
- **`src/task3_hypothesis/hypothesis_tests.py`**: Performs hypothesis testing
- **`src/task4_modeling/train_models.py`**: Trains ML models

### Configuration Files

- **`requirements.txt`**: Python package dependencies
- **`.gitignore`**: Files to exclude from Git
- **`.github/workflows/ci.yml`**: CI/CD pipeline configuration

### Documentation

- **`README.md`**: Project overview and usage
- **`SETUP_GUIDE.md`**: Step-by-step setup instructions
- **`PROJECT_STRUCTURE.md`**: This file

## Output Directories

### Reports Directory

All analysis outputs are saved in the `reports/` directory:

- **`reports/task1_eda/`**:
  - `descriptive_statistics.csv`
  - `missing_values.csv`
  - `correlation_matrix.csv`
  - `loss_ratio_by_province.csv`
  - Various visualization PNG files

- **`reports/task3_hypothesis/`**:
  - `hypothesis_test_summary.csv`
  - `province_risk_analysis.csv`
  - `zipcode_risk_analysis.csv`
  - `zipcode_margin_analysis.csv`
  - `gender_risk_analysis.csv`
  - `hypothesis_test_results.png`

- **`reports/task4_modeling/`**:
  - `model_comparison.csv`
  - `feature_importance_*.csv`
  - `model_comparison.png`
  - `shap_summary_*.png`
  - `shap_bar_*.png`

### Models Directory

Trained models are saved in the `models/` directory:
- `linearregression.joblib`
- `randomforest.joblib`
- `xgboost.joblib`

## Data Flow

1. **Data Input**: CSV file in `data/` directory
2. **Task 1**: EDA analysis → Reports in `reports/task1_eda/`
3. **Task 2**: DVC setup → Data versioning
4. **Task 3**: Hypothesis testing → Reports in `reports/task3_hypothesis/`
5. **Task 4**: Model training → Models in `models/` and reports in `reports/task4_modeling/`

## Git Workflow

The project follows a branch-based workflow:
- `main`: Main branch with merged work
- `task-1`: Branch for Task 1 work
- `task-2`: Branch for Task 2 work
- `task-3`: Branch for Task 3 work
- `task-4`: Branch for Task 4 work

Each task should be completed in its branch and merged via Pull Request.

