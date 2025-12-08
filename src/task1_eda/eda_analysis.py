"""
Exploratory Data Analysis (EDA) for Insurance Risk Analytics

This module performs comprehensive EDA including:
- Data summarization and descriptive statistics
- Data quality assessment
- Univariate and multivariate analysis
- Outlier detection
- Visualization of key insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDAAnalyzer:
    """Class for performing comprehensive EDA on insurance data"""
    
    def __init__(self, data_path: str):
        """
        Initialize EDA Analyzer
        
        Args:
            data_path: Path to the insurance data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.report_path = Path("reports/task1_eda")
        self.report_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the insurance data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def data_summarization(self):
        """Perform data summarization and descriptive statistics"""
        print("\n" + "="*50)
        print("DATA SUMMARIZATION")
        print("="*50)
        
        # Basic info
        print("\n1. Data Structure:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\n2. Data Types:")
        dtype_summary = self.df.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"   {dtype}: {count} columns")
        
        # Descriptive statistics for numerical columns
        print("\n3. Descriptive Statistics (Numerical):")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        desc_stats = self.df[numerical_cols].describe()
        print(desc_stats.to_string())
        
        # Save to file
        desc_stats.to_csv(self.report_path / "descriptive_statistics.csv")
        
        return desc_stats
    
    def data_quality_assessment(self):
        """Assess data quality including missing values"""
        print("\n" + "="*50)
        print("DATA QUALITY ASSESSMENT")
        print("="*50)
        
        # Missing values
        print("\n1. Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string())
            missing_df.to_csv(self.report_path / "missing_values.csv")
        else:
            print("   No missing values found!")
        
        # Duplicates
        print(f"\n2. Duplicate Rows: {self.df.duplicated().sum()}")
        
        return missing_df
    
    def univariate_analysis(self):
        """Perform univariate analysis"""
        print("\n" + "="*50)
        print("UNIVARIATE ANALYSIS")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Numerical distributions
        print(f"\n1. Numerical Variables ({len(numerical_cols)}):")
        for col in numerical_cols[:10]:  # Limit to first 10 for display
            print(f"   {col}: mean={self.df[col].mean():.2f}, std={self.df[col].std():.2f}")
        
        # Categorical distributions
        print(f"\n2. Categorical Variables ({len(categorical_cols)}):")
        for col in categorical_cols[:10]:  # Limit to first 10 for display
            print(f"   {col}: {self.df[col].nunique()} unique values")
        
        # Create histograms for key numerical variables
        key_numerical = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
        key_numerical = [col for col in key_numerical if col in self.df.columns]
        
        if key_numerical:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            for idx, col in enumerate(key_numerical):
                self.df[col].hist(bins=50, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(self.report_path / "numerical_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create bar charts for key categorical variables
        key_categorical = ['Province', 'Gender', 'VehicleType', 'CoverType']
        key_categorical = [col for col in key_categorical if col in self.df.columns]
        
        if key_categorical:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            for idx, col in enumerate(key_categorical):
                if col in self.df.columns:
                    value_counts = self.df[col].value_counts().head(10)
                    value_counts.plot(kind='bar', ax=axes[idx])
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Count')
                    axes[idx].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(self.report_path / "categorical_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def bivariate_analysis(self):
        """Perform bivariate and multivariate analysis"""
        print("\n" + "="*50)
        print("BIVARIATE & MULTIVARIATE ANALYSIS")
        print("="*50)
        
        # Correlation matrix for numerical variables
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        key_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
        key_cols = [col for col in key_cols if col in numerical_cols]
        
        if len(key_cols) > 1:
            corr_matrix = self.df[key_cols].corr()
            print("\n1. Correlation Matrix (Key Financial Variables):")
            print(corr_matrix.to_string())
            corr_matrix.to_csv(self.report_path / "correlation_matrix.csv")
            
            # Visualize correlation matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Correlation Matrix: Key Financial Variables')
            plt.tight_layout()
            plt.savefig(self.report_path / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Monthly changes TotalPremium and TotalClaims as a function of ZipCode
        zipcode_col = None
        for col in ['PostalCode', 'ZipCode', 'Zip', 'Postal']:
            if col in self.df.columns:
                zipcode_col = col
                break
        
        if zipcode_col and 'TransactionMonth' in self.df.columns:
            print("\n2. Monthly Changes in TotalPremium and TotalClaims by ZipCode:")
            
            # Calculate monthly changes by zipcode
            monthly_zipcode_data = self.df.groupby([zipcode_col, 'TransactionMonth']).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).reset_index()
            
            # Calculate month-over-month changes
            monthly_changes = []
            for zipcode in monthly_zipcode_data[zipcode_col].unique():
                zipcode_data = monthly_zipcode_data[monthly_zipcode_data[zipcode_col] == zipcode].sort_values('TransactionMonth')
                zipcode_data['PremiumChange'] = zipcode_data['TotalPremium'].diff()
                zipcode_data['ClaimsChange'] = zipcode_data['TotalClaims'].diff()
                monthly_changes.append(zipcode_data)
            
            if monthly_changes:
                monthly_changes_df = pd.concat(monthly_changes, ignore_index=True)
                monthly_changes_df.to_csv(self.report_path / "monthly_changes_by_zipcode.csv", index=False)
                
                # Scatter plot: Monthly changes TotalPremium vs TotalClaims by ZipCode
                top_zipcodes = monthly_zipcode_data.groupby(zipcode_col)['TotalPremium'].sum().nlargest(10).index
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for zipcode in top_zipcodes:
                    zipcode_changes = monthly_changes_df[monthly_changes_df[zipcode_col] == zipcode]
                    if len(zipcode_changes) > 0:
                        ax.scatter(zipcode_changes['PremiumChange'], zipcode_changes['ClaimsChange'], 
                                 label=f'ZipCode {zipcode}', alpha=0.6, s=100)
                
                ax.set_xlabel('Monthly Change in TotalPremium', fontsize=12, fontweight='bold')
                ax.set_ylabel('Monthly Change in TotalClaims', fontsize=12, fontweight='bold')
                ax.set_title('Monthly Changes: TotalPremium vs TotalClaims by ZipCode', fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.report_path / "monthly_changes_premium_claims_zipcode.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("   Created visualization: Monthly changes by ZipCode")
        
        # Loss Ratio analysis
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / (self.df['TotalPremium'] + 1e-6)
            
            # Overall Loss Ratio
            overall_loss_ratio = self.df['TotalClaims'].sum() / (self.df['TotalPremium'].sum() + 1e-6)
            print(f"\n3. Overall Loss Ratio: {overall_loss_ratio:.4f}")
            
            # Loss Ratio by Province
            if 'Province' in self.df.columns:
                loss_by_province = self.df.groupby('Province').agg({
                    'TotalPremium': 'sum',
                    'TotalClaims': 'sum'
                })
                loss_by_province['LossRatio'] = loss_by_province['TotalClaims'] / (loss_by_province['TotalPremium'] + 1e-6)
                loss_by_province = loss_by_province.sort_values('LossRatio', ascending=False)
                print("\n4. Loss Ratio by Province:")
                print(loss_by_province.to_string())
                loss_by_province.to_csv(self.report_path / "loss_ratio_by_province.csv")
            
            # Loss Ratio by VehicleType
            if 'VehicleType' in self.df.columns:
                loss_by_vehicle = self.df.groupby('VehicleType').agg({
                    'TotalPremium': 'sum',
                    'TotalClaims': 'sum'
                })
                loss_by_vehicle['LossRatio'] = loss_by_vehicle['TotalClaims'] / (loss_by_vehicle['TotalPremium'] + 1e-6)
                loss_by_vehicle = loss_by_vehicle.sort_values('LossRatio', ascending=False)
                print("\n5. Loss Ratio by VehicleType:")
                print(loss_by_vehicle.to_string())
                loss_by_vehicle.to_csv(self.report_path / "loss_ratio_by_vehicletype.csv")
            
            # Loss Ratio by Gender
            if 'Gender' in self.df.columns:
                loss_by_gender = self.df.groupby('Gender').agg({
                    'TotalPremium': 'sum',
                    'TotalClaims': 'sum'
                })
                loss_by_gender['LossRatio'] = loss_by_gender['TotalClaims'] / (loss_by_gender['TotalPremium'] + 1e-6)
                loss_by_gender = loss_by_gender.sort_values('LossRatio', ascending=False)
                print("\n6. Loss Ratio by Gender:")
                print(loss_by_gender.to_string())
                loss_by_gender.to_csv(self.report_path / "loss_ratio_by_gender.csv")
            
            # Vehicle makes/models with highest and lowest claim amounts
            if 'Make' in self.df.columns and 'Model' in self.df.columns:
                vehicle_claims = self.df.groupby(['Make', 'Model']).agg({
                    'TotalClaims': ['sum', 'mean', 'count']
                }).reset_index()
                vehicle_claims.columns = ['Make', 'Model', 'TotalClaims_Sum', 'TotalClaims_Mean', 'Count']
                vehicle_claims = vehicle_claims[vehicle_claims['Count'] >= 5]  # Filter for meaningful sample sizes
                
                highest_claims = vehicle_claims.nlargest(10, 'TotalClaims_Sum')
                lowest_claims = vehicle_claims.nsmallest(10, 'TotalClaims_Sum')
                
                print("\n7. Top 10 Vehicle Makes/Models by Total Claims:")
                print(highest_claims.to_string(index=False))
                print("\n8. Bottom 10 Vehicle Makes/Models by Total Claims:")
                print(lowest_claims.to_string(index=False))
                
                highest_claims.to_csv(self.report_path / "highest_claim_vehicles.csv", index=False)
                lowest_claims.to_csv(self.report_path / "lowest_claim_vehicles.csv", index=False)
            
            # Temporal trends: Claim frequency and severity over time
            if 'TransactionMonth' in self.df.columns:
                self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
                temporal_analysis = self.df.groupby('TransactionMonth').agg({
                    'HasClaim': 'mean',  # Claim frequency
                    'TotalClaims': lambda x: x[x > 0].mean() if (x > 0).any() else 0,  # Claim severity
                    'TotalPremium': 'sum',
                    'TotalClaims': 'sum'
                })
                temporal_analysis.columns = ['ClaimFrequency', 'ClaimSeverity', 'TotalPremium', 'TotalClaims']
                temporal_analysis['LossRatio'] = temporal_analysis['TotalClaims'] / (temporal_analysis['TotalPremium'] + 1e-6)
                
                print("\n9. Temporal Trends (Claim Frequency and Severity):")
                print(temporal_analysis.to_string())
                temporal_analysis.to_csv(self.report_path / "temporal_trends.csv")
    
    def outlier_detection(self):
        """Detect outliers using box plots and IQR method"""
        print("\n" + "="*50)
        print("OUTLIER DETECTION")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        key_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
        key_cols = [col for col in key_cols if col in numerical_cols]
        
        if key_cols:
            # Box plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            for idx, col in enumerate(key_cols):
                self.df.boxplot(column=col, ax=axes[idx])
                axes[idx].set_title(f'Box Plot: {col}')
                axes[idx].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(self.report_path / "outlier_boxplots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # IQR method for outlier detection
            print("\nOutlier Detection (IQR Method):")
            for col in key_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
    
    def create_insight_visualizations(self):
        """Create 3 creative and beautiful plots capturing key insights"""
        print("\n" + "="*50)
        print("CREATIVE VISUALIZATIONS")
        print("="*50)
        
        # Insight 1: Loss Ratio by Province (if available)
        if 'Province' in self.df.columns and 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            province_loss = self.df.groupby('Province').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            })
            province_loss['LossRatio'] = province_loss['TotalClaims'] / (province_loss['TotalPremium'] + 1e-6)
            province_loss = province_loss.sort_values('LossRatio', ascending=False)
            
            bars = ax.barh(province_loss.index, province_loss['LossRatio'], 
                          color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(province_loss))))
            ax.set_xlabel('Loss Ratio', fontsize=12, fontweight='bold')
            ax.set_ylabel('Province', fontsize=12, fontweight='bold')
            ax.set_title('Loss Ratio by Province: Risk Assessment', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.report_path / "insight1_loss_ratio_province.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("   Created: Loss Ratio by Province visualization")
        
        # Insight 2: Temporal trends (if TransactionMonth available)
        if 'TransactionMonth' in self.df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            monthly_data = self.df.groupby('TransactionMonth').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'PolicyID': 'count'
            })
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / (monthly_data['TotalPremium'] + 1e-6)
            
            ax1.plot(monthly_data.index, monthly_data['TotalPremium'], marker='o', label='Total Premium', linewidth=2)
            ax1.plot(monthly_data.index, monthly_data['TotalClaims'], marker='s', label='Total Claims', linewidth=2)
            ax1.set_ylabel('Amount (ZAR)', fontsize=12, fontweight='bold')
            ax1.set_title('Monthly Premium and Claims Trends', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.plot(monthly_data.index, monthly_data['LossRatio'], marker='o', color='red', linewidth=2)
            ax2.set_xlabel('Transaction Month', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Loss Ratio', fontsize=12, fontweight='bold')
            ax2.set_title('Monthly Loss Ratio Trend', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Break-even')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(self.report_path / "insight2_temporal_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("   Created: Temporal trends visualization")
        
        # Insight 3: Risk by Vehicle Type and Gender (if available)
        if 'VehicleType' in self.df.columns and 'Gender' in self.df.columns:
            risk_analysis = self.df.groupby(['VehicleType', 'Gender']).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            })
            risk_analysis['LossRatio'] = risk_analysis['TotalClaims'] / (risk_analysis['TotalPremium'] + 1e-6)
            risk_analysis = risk_analysis.reset_index()
            
            fig, ax = plt.subplots(figsize=(14, 8))
            pivot_data = risk_analysis.pivot(index='VehicleType', columns='Gender', values='LossRatio')
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xlabel('Vehicle Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss Ratio', fontsize=12, fontweight='bold')
            ax.set_title('Risk Profile: Loss Ratio by Vehicle Type and Gender', fontsize=14, fontweight='bold')
            ax.legend(title='Gender')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.report_path / "insight3_vehicle_gender_risk.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("   Created: Vehicle Type and Gender risk analysis")
    
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        self.load_data()
        self.data_summarization()
        self.data_quality_assessment()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.outlier_detection()
        self.create_insight_visualizations()
        
        print("\n" + "="*50)
        print("EDA COMPLETE!")
        print(f"Reports saved to: {self.report_path}")
        print("="*50)


if __name__ == "__main__":
    # Example usage
    data_path = "data/insurance_data.csv"  # Update with actual data path
    
    analyzer = EDAAnalyzer(data_path)
    
    # Check if data file exists
    if Path(data_path).exists():
        analyzer.run_full_eda()
    else:
        print(f"Data file not found at {data_path}")
        print("Please add your insurance data file to the data/ directory")

