"""
XGBoost helper module for forecasting with improved Series data handling
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Union, List, Dict, Any, Optional

def safe_xgboost_data_prep(train_data: Union[pd.DataFrame, pd.Series], 
                           target_col: str,
                           lag_features: int = 3) -> Dict[str, Any]:
    """
    Safely prepare data for XGBoost forecasting by handling both DataFrame and Series inputs.
    
    Args:
        train_data: Either a DataFrame with the target_col or a Series of target values
        target_col: Name of the target column (used if train_data is DataFrame)
        lag_features: Number of lag features to create
        
    Returns:
        Dictionary with prepared data and feature information
    """
    # Check if input is a Series and convert to DataFrame if needed
    if isinstance(train_data, pd.Series):
        # Create a DataFrame with the Series as a column
        df = pd.DataFrame({target_col: train_data})
        df.index = train_data.index
    else:
        # Already a DataFrame
        df = train_data.copy()
        
    # Create date-based features if we have a datetime index
    feature_cols = []
    if isinstance(df.index, pd.DatetimeIndex):
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        feature_cols = ['month', 'quarter', 'year']
    
    # Make sure target column exists in the DataFrame
    if target_col not in df.columns and len(df.columns) > 0:
        # Use the first column as the target if target_col doesn't exist
        original_target = target_col
        target_col = df.columns[0]
        st.info(f"Target column '{original_target}' not found. Using '{target_col}' as the target column.")
    
    # Create lag features if we have enough data
    if len(df) > lag_features:
        for i in range(1, lag_features + 1):
            lag_col = f'{target_col}_lag_{i}'
            df[lag_col] = df[target_col].shift(i)
            feature_cols.append(lag_col)
            
        # Drop rows with NaN values from lag creation
        df = df.dropna()
        
    # Return the prepared data
    return {
        'df': df,
        'feature_cols': feature_cols,
        'target_col': target_col
    }
