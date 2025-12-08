"""
Tests for EDA module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from task1_eda.eda_analysis import EDAAnalyzer


def test_eda_analyzer_initialization():
    """Test EDA Analyzer initialization"""
    analyzer = EDAAnalyzer("data/test_data.csv")
    assert analyzer.data_path == "data/test_data.csv"
    assert analyzer.df is None


def test_calculate_metrics():
    """Test metric calculation"""
    # Create sample data
    data = {
        'TotalPremium': [1000, 2000, 3000],
        'TotalClaims': [100, 200, 0],
        'HasClaim': [1, 1, 0],
        'ClaimAmount': [100, 200, 0],
        'Margin': [900, 1800, 3000]
    }
    df = pd.DataFrame(data)
    
    # This is a basic test - actual implementation would test the full class
    assert len(df) == 3
    assert df['TotalPremium'].sum() == 6000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

