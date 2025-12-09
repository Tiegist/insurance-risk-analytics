# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project is part of the **10 Academy: Artificial Intelligence Mastery** program, focusing on **End-to-End Insurance Risk Analytics & Predictive Modeling**. The objective is to analyze historical insurance claim data to help optimize marketing strategy and discover "low-risk" targets for premium reduction, creating opportunities to attract new clients.

## Business Objective

Your employer, AlphaCare Insurance Solutions (ACIS), is committed to developing cutting-edge risk and predictive analytics in the area of car insurance planning and marketing in South Africa. You have recently joined the data analytics team as a marketing analytics engineer, and your first project is to analyse historical insurance claim data. The objective of your analyses is to help optimise the marketing strategy and discover "low-risk" targets for which the premium could be reduced, thereby creating an opportunity to attract new clients.

The historical data is from February 2014 to August 2015.

## Project Structure

```
insurance-risk-analytics/
├── data/                      # Data files (tracked with DVC)
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── task1_eda/            # Task 1: EDA and Statistical Analysis
│   ├── task2_dvc/            # Task 2: Data Version Control
│   ├── task3_hypothesis/     # Task 3: A/B Hypothesis Testing
│   └── task4_modeling/       # Task 4: Machine Learning Models
├── reports/                   # Generated reports and visualizations
├── tests/                     # Unit tests
├── .github/                   # GitHub Actions workflows
│   └── workflows/
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Tasks

### Task 1: Git/GitHub & EDA
**1.1 Git and GitHub**
- Create a git repository with a good README
- Git version control
- CI/CD with GitHub Actions

**1.2 Project Planning - EDA & Stats**
- Data Understanding
- Exploratory Data Analysis (EDA)
  - Overall Loss Ratio and variation by Province, VehicleType, and Gender
  - Distributions of key financial variables
  - Temporal trends (claim frequency/severity over 18-month period)
  - Vehicle makes/models with highest and lowest claim amounts
  - Monthly changes TotalPremium and TotalClaims as function of ZipCode
- Statistical thinking with suitable distributions and plots

### Task 2: Data Version Control (DVC)
- Install DVC
- Initialize DVC
- Set up local remote storage
- Add data to DVC
- Commit and push data versions

### Task 3: A/B Hypothesis Testing
Accept or reject the following Null Hypotheses:
- H₀: There are no risk differences across provinces
- H₀: There are no risk differences between zip codes
- H₀: There is no significant margin (profit) difference between zip codes
- H₀: There is no significant risk difference between Women and Men

Metrics: Claim Frequency and Claim Severity

### Task 4: Machine Learning & Statistical Modeling
- **For each zipcode, fit a linear regression model that predicts the total claims**
- Claim Severity Prediction (Risk Model): For policies with claims > 0
- Premium Optimization Model
- Implement Linear Regression, Random Forests, and XGBoost
- Model evaluation (RMSE, R²)
- Feature importance analysis using SHAP or LIME

## Key Metrics

- **Loss Ratio**: TotalClaims / TotalPremium
- **Claim Frequency**: Proportion of policies with at least one claim
- **Claim Severity**: Average amount of a claim, given a claim occurred
- **Margin**: TotalPremium - TotalClaims

## Data Structure

The dataset includes:
- **Policy Information**: UnderwrittenCoverID, PolicyID, TransactionMonth
- **Client Information**: Gender, MaritalStatus, Citizenship, LegalType, etc.
- **Location Information**: Country, Province, PostalCode, MainCrestaZone, SubCrestaZone
- **Vehicle Information**: Make, Model, RegistrationYear, VehicleType, etc.
- **Plan Information**: SumInsured, CalculatedPremiumPerTerm, CoverType, etc.
- **Financial Information**: TotalPremium, TotalClaims

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- DVC

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd insurance-risk-analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
```

## Usage

### Quick Start: Run All Analyses and Generate Reports
```bash
# Place your data file in data/ directory (e.g., insurance_data.csv)
python run_analysis_and_generate_reports.py
```
This will run all tasks and automatically generate reports with actual findings.

### Run Tasks Individually

#### Task 1: EDA
```bash
python src/task1_eda/eda_analysis.py
```

#### Task 2: DVC Setup
```bash
python src/task2_dvc/dvc_setup.py
dvc add data/insurance_data.csv
dvc push
```

#### Task 3: Hypothesis Testing
```bash
python src/task3_hypothesis/hypothesis_tests.py
```

#### Task 4: Modeling
```bash
python src/task4_modeling/train_models.py
```


## Deliverables

- **Interim Submission** (Sunday, 07 Dec 2025): Task 1 & 2
- **Final Submission** (Tuesday, 09 Dec 2025): Complete project with final report

## References

- [Insurance Analytics Resources](https://www.fsrao.ca/media/11501/download)
- [A/B Testing Guide](https://medium.com/tiket-com/a-b-testing-hypothesis-testing-f9624ea5580e)
- [DVC Documentation](https://dvc.org/)
- [Statistical Modeling](https://www.coursera.org/articles/statistical-modeling)

## Contributors

- Marketing Analytics Engineer Team

## License

This project is part of the 10 Academy AI Mastery program.

