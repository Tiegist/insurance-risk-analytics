"""
A/B Hypothesis Testing for Insurance Risk Analytics

This module performs statistical hypothesis testing to validate or reject
key hypotheses about risk drivers:
- Risk differences across provinces
- Risk differences between zip codes
- Margin differences between zip codes
- Risk differences between genders
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HypothesisTester:
    """Class for performing A/B hypothesis tests on insurance data"""
    
    def __init__(self, data_path: str):
        """
        Initialize Hypothesis Tester
        
        Args:
            data_path: Path to the insurance data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.report_path = Path("reports/task3_hypothesis")
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def load_data(self):
        """Load the insurance data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Calculate key metrics
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
            self.df['ClaimAmount'] = self.df['TotalClaims']
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            self.df['LossRatio'] = self.df['TotalClaims'] / (self.df['TotalPremium'] + 1e-6)
        
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def calculate_metrics(self, group_data):
        """Calculate key metrics for a group"""
        metrics = {
            'count': len(group_data),
            'claim_frequency': group_data['HasClaim'].mean() if 'HasClaim' in group_data.columns else 0,
            'claim_severity': group_data[group_data['HasClaim'] == 1]['ClaimAmount'].mean() if 'HasClaim' in group_data.columns else 0,
            'total_premium': group_data['TotalPremium'].sum() if 'TotalPremium' in group_data.columns else 0,
            'total_claims': group_data['TotalClaims'].sum() if 'TotalClaims' in group_data.columns else 0,
            'avg_margin': group_data['Margin'].mean() if 'Margin' in group_data.columns else 0,
            'loss_ratio': (group_data['TotalClaims'].sum() / (group_data['TotalPremium'].sum() + 1e-6)) if 'TotalPremium' in group_data.columns else 0
        }
        return metrics
    
    def test_province_risk_differences(self, alpha=0.05):
        """
        Test H₀: There are no risk differences across provinces
        
        Uses chi-squared test for claim frequency and t-test for claim severity
        """
        print("\n" + "="*50)
        print("HYPOTHESIS TEST 1: Risk Differences Across Provinces")
        print("="*50)
        
        if 'Province' not in self.df.columns:
            print("✗ Province column not found in data")
            return None
        
        # Group by province
        provinces = self.df['Province'].unique()
        if len(provinces) < 2:
            print("✗ Need at least 2 provinces for comparison")
            return None
        
        print(f"\nTesting {len(provinces)} provinces...")
        
        # Test 1: Claim Frequency (Chi-squared test)
        contingency_table = pd.crosstab(self.df['Province'], self.df['HasClaim'])
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n1. Claim Frequency Test (Chi-squared):")
        print(f"   Chi-squared statistic: {chi2:.4f}")
        print(f"   p-value: {p_value_freq:.6f}")
        print(f"   Degrees of freedom: {dof}")
        
        if p_value_freq < alpha:
            print(f"   ✓ REJECT H₀ (p < {alpha}): There ARE significant risk differences across provinces")
        else:
            print(f"   ✗ FAIL TO REJECT H₀ (p >= {alpha}): No significant risk differences across provinces")
        
        # Test 2: Claim Severity (ANOVA or Kruskal-Wallis)
        province_claims = [self.df[(self.df['Province'] == prov) & (self.df['HasClaim'] == 1)]['ClaimAmount'].values 
                          for prov in provinces]
        province_claims = [arr for arr in province_claims if len(arr) > 0]
        
        if len(province_claims) >= 2:
            # Use Kruskal-Wallis for non-parametric test (more robust to outliers)
            h_stat, p_value_sev = stats.kruskal(*province_claims)
            
            print(f"\n2. Claim Severity Test (Kruskal-Wallis):")
            print(f"   H-statistic: {h_stat:.4f}")
            print(f"   p-value: {p_value_sev:.6f}")
            
            if p_value_sev < alpha:
                print(f"   ✓ REJECT H₀ (p < {alpha}): There ARE significant severity differences across provinces")
            else:
                print(f"   ✗ FAIL TO REJECT H₀ (p >= {alpha}): No significant severity differences across provinces")
        
        # Summary by province
        print("\n3. Summary by Province:")
        province_summary = self.df.groupby('Province').apply(self.calculate_metrics).reset_index()
        province_summary = pd.DataFrame(province_summary[0].tolist())
        province_summary['Province'] = self.df.groupby('Province').apply(lambda x: x.name).values
        print(province_summary.to_string(index=False))
        province_summary.to_csv(self.report_path / "province_risk_analysis.csv", index=False)
        
        result = {
            'hypothesis': 'Risk differences across provinces',
            'test_type': 'Chi-squared (frequency) + Kruskal-Wallis (severity)',
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev if len(province_claims) >= 2 else None,
            'rejected': (p_value_freq < alpha) or (p_value_sev < alpha if len(province_claims) >= 2 else False),
            'alpha': alpha
        }
        self.results.append(result)
        
        return result
    
    def test_zipcode_risk_differences(self, alpha=0.05, top_n=10):
        """
        Test H₀: There are no risk differences between zip codes
        
        Compares top N zip codes by volume
        """
        print("\n" + "="*50)
        print("HYPOTHESIS TEST 2: Risk Differences Between Zip Codes")
        print("="*50)
        
        zipcode_col = None
        for col in ['PostalCode', 'ZipCode', 'Zip', 'Postal']:
            if col in self.df.columns:
                zipcode_col = col
                break
        
        if zipcode_col is None:
            print("✗ Zip code column not found in data")
            return None
        
        # Get top N zip codes by volume
        zipcode_counts = self.df[zipcode_col].value_counts().head(top_n)
        top_zipcodes = zipcode_counts.index.tolist()
        
        print(f"\nComparing top {len(top_zipcodes)} zip codes by volume...")
        
        # Test claim frequency
        zipcode_claims = []
        zipcode_no_claims = []
        
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df[zipcode_col] == zipcode]
            zipcode_claims.append(zipcode_data['HasClaim'].sum())
            zipcode_no_claims.append((zipcode_data['HasClaim'] == 0).sum())
        
        contingency_table = pd.DataFrame({
            'Claims': zipcode_claims,
            'No_Claims': zipcode_no_claims
        }, index=top_zipcodes)
        
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n1. Claim Frequency Test (Chi-squared):")
        print(f"   Chi-squared statistic: {chi2:.4f}")
        print(f"   p-value: {p_value_freq:.6f}")
        
        if p_value_freq < alpha:
            print(f"   ✓ REJECT H₀ (p < {alpha}): There ARE significant risk differences between zip codes")
        else:
            print(f"   ✗ FAIL TO REJECT H₀ (p >= {alpha}): No significant risk differences between zip codes")
        
        # Summary by zip code
        print("\n2. Summary by Top Zip Codes:")
        zipcode_summary = []
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df[zipcode_col] == zipcode]
            metrics = self.calculate_metrics(zipcode_data)
            metrics[zipcode_col] = zipcode
            zipcode_summary.append(metrics)
        
        zipcode_df = pd.DataFrame(zipcode_summary)
        print(zipcode_df.to_string(index=False))
        zipcode_df.to_csv(self.report_path / "zipcode_risk_analysis.csv", index=False)
        
        result = {
            'hypothesis': 'Risk differences between zip codes',
            'test_type': 'Chi-squared',
            'p_value': p_value_freq,
            'rejected': p_value_freq < alpha,
            'alpha': alpha
        }
        self.results.append(result)
        
        return result
    
    def test_zipcode_margin_differences(self, alpha=0.05, top_n=10):
        """
        Test H₀: There is no significant margin (profit) difference between zip codes
        """
        print("\n" + "="*50)
        print("HYPOTHESIS TEST 3: Margin Differences Between Zip Codes")
        print("="*50)
        
        zipcode_col = None
        for col in ['PostalCode', 'ZipCode', 'Zip', 'Postal']:
            if col in self.df.columns:
                zipcode_col = col
                break
        
        if zipcode_col is None:
            print("✗ Zip code column not found in data")
            return None
        
        # Get top N zip codes by volume
        zipcode_counts = self.df[zipcode_col].value_counts().head(top_n)
        top_zipcodes = zipcode_counts.index.tolist()
        
        print(f"\nComparing margins across top {len(top_zipcodes)} zip codes...")
        
        # Perform ANOVA or Kruskal-Wallis test
        zipcode_margins = [self.df[self.df[zipcode_col] == zipcode]['Margin'].values 
                          for zipcode in top_zipcodes]
        zipcode_margins = [arr for arr in zipcode_margins if len(arr) > 0]
        
        if len(zipcode_margins) >= 2:
            # Use Kruskal-Wallis for non-parametric test
            h_stat, p_value = stats.kruskal(*zipcode_margins)
            
            print(f"\n1. Margin Difference Test (Kruskal-Wallis):")
            print(f"   H-statistic: {h_stat:.4f}")
            print(f"   p-value: {p_value:.6f}")
            
            if p_value < alpha:
                print(f"   ✓ REJECT H₀ (p < {alpha}): There ARE significant margin differences between zip codes")
            else:
                print(f"   ✗ FAIL TO REJECT H₀ (p >= {alpha}): No significant margin differences between zip codes")
        
        # Summary by zip code
        print("\n2. Margin Summary by Top Zip Codes:")
        zipcode_margins_summary = []
        for zipcode in top_zipcodes:
            zipcode_data = self.df[self.df[zipcode_col] == zipcode]
            metrics = self.calculate_metrics(zipcode_data)
            metrics[zipcode_col] = zipcode
            zipcode_margins_summary.append(metrics)
        
        zipcode_margin_df = pd.DataFrame(zipcode_margins_summary)
        print(zipcode_margin_df[['PostalCode', 'avg_margin', 'total_premium', 'total_claims', 'loss_ratio']].to_string(index=False))
        zipcode_margin_df.to_csv(self.report_path / "zipcode_margin_analysis.csv", index=False)
        
        result = {
            'hypothesis': 'Margin differences between zip codes',
            'test_type': 'Kruskal-Wallis',
            'p_value': p_value if len(zipcode_margins) >= 2 else None,
            'rejected': p_value < alpha if len(zipcode_margins) >= 2 else False,
            'alpha': alpha
        }
        self.results.append(result)
        
        return result
    
    def test_gender_risk_differences(self, alpha=0.05):
        """
        Test H₀: There is no significant risk difference between Women and Men
        """
        print("\n" + "="*50)
        print("HYPOTHESIS TEST 4: Risk Differences Between Genders")
        print("="*50)
        
        if 'Gender' not in self.df.columns:
            print("✗ Gender column not found in data")
            return None
        
        # Filter to only Male and Female (if other values exist)
        gender_data = self.df[self.df['Gender'].isin(['Male', 'Female', 'M', 'F'])].copy()
        
        if len(gender_data) == 0:
            print("✗ No valid gender data found")
            return None
        
        # Normalize gender values
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        gender_data['Gender'] = gender_data['Gender'].map(gender_mapping).fillna(gender_data['Gender'])
        
        genders = gender_data['Gender'].unique()
        if len(genders) < 2:
            print("✗ Need both Male and Female data for comparison")
            return None
        
        print(f"\nComparing genders: {', '.join(genders)}...")
        
        # Test 1: Claim Frequency (Chi-squared test)
        contingency_table = pd.crosstab(gender_data['Gender'], gender_data['HasClaim'])
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n1. Claim Frequency Test (Chi-squared):")
        print(f"   Chi-squared statistic: {chi2:.4f}")
        print(f"   p-value: {p_value_freq:.6f}")
        
        if p_value_freq < alpha:
            print(f"   ✓ REJECT H₀ (p < {alpha}): There ARE significant risk differences between genders")
        else:
            print(f"   ✗ FAIL TO REJECT H₀ (p >= {alpha}): No significant risk differences between genders")
        
        # Test 2: Claim Severity (Mann-Whitney U test - non-parametric)
        male_claims = gender_data[(gender_data['Gender'] == 'Male') & (gender_data['HasClaim'] == 1)]['ClaimAmount'].values
        female_claims = gender_data[(gender_data['Gender'] == 'Female') & (gender_data['HasClaim'] == 1)]['ClaimAmount'].values
        
        if len(male_claims) > 0 and len(female_claims) > 0:
            u_stat, p_value_sev = mannwhitneyu(male_claims, female_claims, alternative='two-sided')
            
            print(f"\n2. Claim Severity Test (Mann-Whitney U):")
            print(f"   U-statistic: {u_stat:.4f}")
            print(f"   p-value: {p_value_sev:.6f}")
            
            if p_value_sev < alpha:
                print(f"   ✓ REJECT H₀ (p < {alpha}): There ARE significant severity differences between genders")
            else:
                print(f"   ✗ FAIL TO REJECT H₀ (p >= {alpha}): No significant severity differences between genders")
        
        # Summary by gender
        print("\n3. Summary by Gender:")
        gender_summary = gender_data.groupby('Gender').apply(self.calculate_metrics).reset_index()
        gender_summary = pd.DataFrame(gender_summary[0].tolist())
        gender_summary['Gender'] = gender_data.groupby('Gender').apply(lambda x: x.name).values
        print(gender_summary.to_string(index=False))
        gender_summary.to_csv(self.report_path / "gender_risk_analysis.csv", index=False)
        
        result = {
            'hypothesis': 'Risk differences between genders',
            'test_type': 'Chi-squared (frequency) + Mann-Whitney U (severity)',
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev if len(male_claims) > 0 and len(female_claims) > 0 else None,
            'rejected': (p_value_freq < alpha) or (p_value_sev < alpha if len(male_claims) > 0 and len(female_claims) > 0 else False),
            'alpha': alpha
        }
        self.results.append(result)
        
        return result
    
    def generate_report(self):
        """Generate a summary report of all hypothesis tests"""
        print("\n" + "="*50)
        print("HYPOTHESIS TESTING SUMMARY REPORT")
        print("="*50)
        
        if not self.results:
            print("No test results available")
            return
        
        results_df = pd.DataFrame(self.results)
        print("\n" + results_df.to_string(index=False))
        results_df.to_csv(self.report_path / "hypothesis_test_summary.csv", index=False)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        hypotheses = [r['hypothesis'] for r in self.results]
        p_values = [r.get('p_value', r.get('p_value_frequency', 0)) for r in self.results]
        rejected = [r['rejected'] for r in self.results]
        
        colors = ['red' if r else 'green' for r in rejected]
        bars = ax.barh(range(len(hypotheses)), p_values, color=colors, alpha=0.7)
        ax.axvline(x=0.05, color='black', linestyle='--', label='α = 0.05')
        ax.set_yticks(range(len(hypotheses)))
        ax.set_yticklabels([h[:40] + '...' if len(h) > 40 else h for h in hypotheses])
        ax.set_xlabel('p-value', fontsize=12, fontweight='bold')
        ax.set_title('Hypothesis Test Results', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.report_path / "hypothesis_test_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nReport saved to: {self.report_path}")
    
    def run_all_tests(self, alpha=0.05):
        """Run all hypothesis tests"""
        print("="*50)
        print("A/B HYPOTHESIS TESTING")
        print("="*50)
        
        self.load_data()
        self.test_province_risk_differences(alpha)
        self.test_zipcode_risk_differences(alpha)
        self.test_zipcode_margin_differences(alpha)
        self.test_gender_risk_differences(alpha)
        self.generate_report()
        
        print("\n" + "="*50)
        print("HYPOTHESIS TESTING COMPLETE!")
        print(f"Reports saved to: {self.report_path}")
        print("="*50)


if __name__ == "__main__":
    # Example usage
    data_path = "data/insurance_data.csv"  # Update with actual data path
    
    tester = HypothesisTester(data_path)
    
    # Check if data file exists
    if Path(data_path).exists():
        tester.run_all_tests(alpha=0.05)
    else:
        print(f"Data file not found at {data_path}")
        print("Please add your insurance data file to the data/ directory")

