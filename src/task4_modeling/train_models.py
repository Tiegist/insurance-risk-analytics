"""
Machine Learning Models for Insurance Risk Analytics

This module implements:
1. Claim Severity Prediction (for policies with claims)
2. Premium Optimization Model
3. Model evaluation and comparison
4. Feature importance analysis using SHAP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
import joblib

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class InsuranceModelTrainer:
    """Class for training and evaluating insurance risk models"""
    
    def __init__(self, data_path: str):
        """
        Initialize Model Trainer
        
        Args:
            data_path: Path to the insurance data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.report_path = Path("reports/task4_modeling")
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.models_path = Path("models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def load_data(self):
        """Load and prepare the insurance data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def prepare_features(self, task='severity'):
        """
        Prepare features for modeling
        
        Args:
            task: 'severity' for claim severity prediction, 'premium' for premium optimization
        """
        print(f"\nPreparing features for {task} prediction...")
        
        df = self.df.copy()
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Feature engineering
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)
            df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        
        if 'RegistrationYear' in df.columns:
            df['VehicleAge'] = 2024 - df['RegistrationYear']  # Approximate age
        
        # Select features based on task
        if task == 'severity':
            # For claim severity, only use policies with claims
            df = df[df['TotalClaims'] > 0].copy()
            target = 'TotalClaims'
        else:
            # For premium optimization
            target = 'CalculatedPremiumPerTerm' if 'CalculatedPremiumPerTerm' in df.columns else 'TotalPremium'
        
        # Select relevant features
        feature_cols = []
        
        # Client features
        client_features = ['Gender', 'MaritalStatus', 'Citizenship', 'LegalType', 'IsVATRegistered']
        feature_cols.extend([col for col in client_features if col in df.columns])
        
        # Location features
        location_features = ['Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone']
        feature_cols.extend([col for col in location_features if col in df.columns])
        
        # Vehicle features
        vehicle_features = ['VehicleType', 'Make', 'Model', 'Bodytype', 'Cylinders', 
                          'Cubiccapacity', 'Kilowatts', 'NumberOfDoors', 'RegistrationYear']
        feature_cols.extend([col for col in vehicle_features if col in df.columns])
        
        # Plan features
        plan_features = ['CoverType', 'CoverCategory', 'CoverGroup', 'SumInsured', 'ExcessSelected']
        feature_cols.extend([col for col in plan_features if col in df.columns])
        
        # Engineered features
        if 'VehicleAge' in df.columns:
            feature_cols.append('VehicleAge')
        if 'LossRatio' in df.columns and task != 'severity':
            feature_cols.append('LossRatio')
        
        # Remove target from features
        feature_cols = [col for col in feature_cols if col != target]
        
        # Separate numerical and categorical features
        numerical_features = [col for col in feature_cols if col in numerical_cols]
        categorical_features = [col for col in feature_cols if col in categorical_cols]
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        print(f"   Features selected: {len(feature_cols)}")
        print(f"   Numerical features: {len(numerical_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        print(f"   Samples: {len(X)}")
        
        return X, y, numerical_features, categorical_features
    
    def encode_features(self, X, numerical_features, categorical_features, fit=True):
        """Encode categorical features and scale numerical features"""
        X_encoded = X.copy()
        
        # Encode categorical features
        if fit:
            self.encoders = {}
            for col in categorical_features:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.encoders[col] = le
        else:
            for col in categorical_features:
                if col in self.encoders:
                    # Handle unseen categories
                    X_encoded[col] = X_encoded[col].astype(str).map(
                        lambda x: self.encoders[col].transform([x])[0] 
                        if x in self.encoders[col].classes_ 
                        else -1
                    )
        
        # Scale numerical features
        if fit:
            self.scalers['numerical'] = StandardScaler()
            X_encoded[numerical_features] = self.scalers['numerical'].fit_transform(X_encoded[numerical_features])
        else:
            if 'numerical' in self.scalers:
                X_encoded[numerical_features] = self.scalers['numerical'].transform(X_encoded[numerical_features])
        
        return X_encoded
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        print("\nTraining Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"   Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
        print(f"   Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
        
        self.models['LinearRegression'] = model
        
        return {
            'model': 'LinearRegression',
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"   Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
        print(f"   Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
        
        self.models['RandomForest'] = model
        
        return {
            'model': 'RandomForest',
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"   Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
        print(f"   Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
        
        self.models['XGBoost'] = model
        
        return {
            'model': 'XGBoost',
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def analyze_feature_importance(self, X_test, model_name='XGBoost', top_n=10):
        """Analyze feature importance using SHAP"""
        print(f"\nAnalyzing feature importance using SHAP for {model_name}...")
        
        if model_name not in self.models:
            print(f"   Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Use a sample for SHAP (can be slow for large datasets)
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=top_n)
            plt.tight_layout()
            plt.savefig(self.report_path / f"shap_summary_{model_name.lower()}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot of mean SHAP values
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=top_n)
            plt.tight_layout()
            plt.savefig(self.report_path / f"shap_bar_{model_name.lower()}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Get feature importance values
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False).head(top_n)
            
            print(f"\n   Top {top_n} Most Important Features:")
            print(feature_importance.to_string(index=False))
            feature_importance.to_csv(self.report_path / f"feature_importance_{model_name.lower()}.csv", index=False)
            
            return feature_importance
            
        except Exception as e:
            print(f"   Error in SHAP analysis: {str(e)}")
            # Fallback to built-in feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(top_n)
                
                print(f"\n   Top {top_n} Most Important Features (built-in):")
                print(feature_importance.to_string(index=False))
                feature_importance.to_csv(self.report_path / f"feature_importance_{model_name.lower()}.csv", index=False)
                
                return feature_importance
            return None
    
    def compare_models(self, results):
        """Compare model performance"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))
        results_df.to_csv(self.report_path / "model_comparison.csv", index=False)
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = results_df['model'].values
        metrics = ['test_rmse', 'test_r2', 'test_mae']
        metric_labels = ['RMSE (Lower is Better)', 'R² (Higher is Better)', 'MAE (Lower is Better)']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = results_df[metric].values
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = axes[idx].bar(models, values, color=colors, alpha=0.7)
            axes[idx].set_ylabel(label, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'Model Comparison: {label}', fontsize=14, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.report_path / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find best model
        best_model_idx = results_df['test_r2'].idxmax()
        best_model = results_df.loc[best_model_idx, 'model']
        print(f"\n✓ Best Model (by R²): {best_model}")
        print(f"   Test R²: {results_df.loc[best_model_idx, 'test_r2']:.4f}")
        print(f"   Test RMSE: {results_df.loc[best_model_idx, 'test_rmse']:.2f}")
        
        return best_model
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        for name, model in self.models.items():
            model_path = self.models_path / f"{name.lower()}.joblib"
            joblib.dump(model, model_path)
            print(f"   Saved: {model_path}")
    
    def train_claim_severity_model(self):
        """Train model to predict claim severity"""
        print("="*50)
        print("CLAIM SEVERITY PREDICTION MODEL")
        print("="*50)
        
        X, y, numerical_features, categorical_features = self.prepare_features(task='severity')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Encode features
        X_train_encoded = self.encode_features(X_train, numerical_features, categorical_features, fit=True)
        X_test_encoded = self.encode_features(X_test, numerical_features, categorical_features, fit=False)
        
        # Train models
        results = []
        results.append(self.train_linear_regression(X_train_encoded, y_train, X_test_encoded, y_test))
        results.append(self.train_random_forest(X_train_encoded, y_train, X_test_encoded, y_test))
        results.append(self.train_xgboost(X_train_encoded, y_train, X_test_encoded, y_test))
        
        # Compare models
        best_model = self.compare_models(results)
        
        # Feature importance analysis
        self.analyze_feature_importance(X_test_encoded, model_name=best_model)
        
        # Save models
        self.save_models()
        
        self.results['severity'] = results
        
        return results
    
    def train_premium_optimization_model(self):
        """Train model to predict optimal premium"""
        print("\n" + "="*50)
        print("PREMIUM OPTIMIZATION MODEL")
        print("="*50)
        
        X, y, numerical_features, categorical_features = self.prepare_features(task='premium')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Encode features
        X_train_encoded = self.encode_features(X_train, numerical_features, categorical_features, fit=True)
        X_test_encoded = self.encode_features(X_test, numerical_features, categorical_features, fit=False)
        
        # Train models
        results = []
        results.append(self.train_linear_regression(X_train_encoded, y_train, X_test_encoded, y_test))
        results.append(self.train_random_forest(X_train_encoded, y_train, X_test_encoded, y_test))
        results.append(self.train_xgboost(X_train_encoded, y_train, X_test_encoded, y_test))
        
        # Compare models
        best_model = self.compare_models(results)
        
        # Feature importance analysis
        self.analyze_feature_importance(X_test_encoded, model_name=best_model)
        
        # Save models
        self.save_models()
        
        self.results['premium'] = results
        
        return results
    
    def run_all_modeling(self):
        """Run complete modeling pipeline"""
        print("="*50)
        print("MACHINE LEARNING MODELING")
        print("="*50)
        
        self.load_data()
        self.train_claim_severity_model()
        self.train_premium_optimization_model()
        
        print("\n" + "="*50)
        print("MODELING COMPLETE!")
        print(f"Reports saved to: {self.report_path}")
        print(f"Models saved to: {self.models_path}")
        print("="*50)


if __name__ == "__main__":
    # Example usage
    data_path = "data/insurance_data.csv"  # Update with actual data path
    
    trainer = InsuranceModelTrainer(data_path)
    
    # Check if data file exists
    if Path(data_path).exists():
        trainer.run_all_modeling()
    else:
        print(f"Data file not found at {data_path}")
        print("Please add your insurance data file to the data/ directory")

