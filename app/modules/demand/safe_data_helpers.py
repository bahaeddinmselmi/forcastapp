"""
Safe data handling helpers for forecast models
"""

import pandas as pd
import numpy as np
import streamlit as st

def safely_get_target_data(train_data, target_col, display_warnings=True):
    """
    Safely extracts target column data from a dataframe or series for forecasting.
    
    Args:
        train_data: DataFrame or Series containing the target data
        target_col: Target column name to use (if applicable)
        display_warnings: Whether to show Streamlit warning messages
        
    Returns:
        Series or array of target data
    """
    # Case 1: If already a Series, just return it
    if isinstance(train_data, pd.Series):
        return train_data
        
    # Case 2: DataFrame handling
    if isinstance(train_data, pd.DataFrame):
        # Check if target_col exists in DataFrame
        if target_col in train_data.columns:
            return train_data[target_col]
            
        # If target column doesn't exist but we have a single column
        if train_data.shape[1] == 1:
            actual_col = train_data.columns[0]
            if display_warnings:
                st.info(f"Target column '{target_col}' not found. Using '{actual_col}' instead.")
            return train_data[actual_col]
            
        # If we have multiple columns, look for numeric ones
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            actual_col = numeric_cols[0]
            if display_warnings:
                st.warning(f"Target column '{target_col}' not found. Using '{actual_col}' as fallback.")
            return train_data[actual_col]
            
        # Last resort: try common names
        for common_name in ['value', 'sales', 'amount', 'quantity', 'demand']:
            if common_name in train_data.columns:
                if display_warnings:
                    st.warning(f"Target column '{target_col}' not found. Using '{common_name}' as fallback.")
                return train_data[common_name]
                
        # If all else fails, raise a more informative error
        available_cols = ", ".join(train_data.columns)
        raise KeyError(f"Could not find target column '{target_col}' or suitable numeric column for forecasting. Available columns: {available_cols}")
        
    # Case 3: numpy array or other type - just return it
    return train_data
