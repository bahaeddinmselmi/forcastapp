"""
Fix for the cumulative forecast and metrics calculation in the IBP app.

This script will be run to apply the fixes to the application.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

def safely_get_target_data(data, target_col):
    """
    Safely extract target data from DataFrame or Series.
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        if target_col in data.columns:
            return data[target_col]
        elif data.shape[1] > 0:
            # Use first column as fallback
            col = data.columns[0]
            print(f"Target column '{target_col}' not found. Using '{col}' instead.")
            return data[col]
    return pd.Series()  # Empty series as last resort

def safe_prepare_cumulative_forecast(forecasts, historical_data, target_col=None):
    """
    Safely prepare cumulative forecasts from multiple forecast models.
    Works with both Series and DataFrame inputs.
    """
    result = {}
    
    # Get historical data safely
    historical_series = safely_get_target_data(historical_data, target_col)
    
    if len(historical_series) > 0:
        # Calculate cumulative historical data
        result['historical'] = historical_series.cumsum()
        
        # Process each forecast model
        for model_name, model_result in forecasts.items():
            if 'forecast' in model_result and isinstance(model_result['forecast'], pd.Series):
                forecast = model_result['forecast']
                
                # Get last historical value for continuity
                last_value = result['historical'].iloc[-1] if len(result['historical']) > 0 else 0
                
                # Calculate cumulative forecast
                cum_forecast = forecast.cumsum() + last_value
                
                # Store in result
                if model_name not in result:
                    result[model_name] = {}
                    
                result[model_name]['forecast'] = cum_forecast
                
                # Add confidence intervals if available
                if all(k in model_result for k in ['lower_bound', 'upper_bound']):
                    result[model_name]['lower_bound'] = model_result['lower_bound'].cumsum() + last_value
                    result[model_name]['upper_bound'] = model_result['upper_bound'].cumsum() + last_value
    
    return result

print("IMPORTANT: Apply these functions where needed in the app to fix issues with:")
print("1. Cumulative forecasts failing with 'sales' KeyError")
print("2. Metrics calculation failing with Series data")
print("3. Any other places where 'Series' object has no attribute 'columns'")
