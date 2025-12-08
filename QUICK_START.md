# Quick Start Guide

Get started with the Insurance Risk Analytics project in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Add Your Data

Place your insurance data CSV file in the `data/` directory and name it `insurance_data.csv`.

## 3. Run Tasks

### Option A: Run All Tasks at Once
```bash
python run_all_tasks.py
```

### Option B: Run Tasks Individually

**Task 1: EDA**
```bash
python src/task1_eda/eda_analysis.py
```

**Task 2: DVC Setup**
```bash
python src/task2_dvc/dvc_setup.py
dvc add data/insurance_data.csv
```

**Task 3: Hypothesis Testing**
```bash
python src/task3_hypothesis/hypothesis_tests.py
```

**Task 4: Machine Learning**
```bash
python src/task4_modeling/train_models.py
```

## 4. View Results

Check the `reports/` directory for all generated analysis and visualizations.

## 5. Git Workflow

```bash
# Create branch for Task 1
git checkout -b task-1

# Make changes and commit
git add .
git commit -m "feat: complete task 1"

# Push and create PR
git push origin task-1
```

## Need Help?

- See `SETUP_GUIDE.md` for detailed setup instructions
- See `README.md` for project overview
- See `PROJECT_STRUCTURE.md` for project organization

