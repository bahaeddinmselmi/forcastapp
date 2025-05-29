"""
Helper functions for cumulative forecasts with safe Series handling
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Union, Optional

def safely_get_cumulative_data(data: Union[pd.DataFrame, pd.Series], 
                               target_col: Optional[str] = None) -> pd.Series:
    """
    Safely extract and cumulate data from DataFrame or Series.
    
    Args:
        data: DataFrame or Series with the data
        target_col: Column name if data is a DataFrame
        
    Returns:
        Cumulative Series
    """
    # Get the target data
    if isinstance(data, pd.DataFrame):
        if target_col is not None and target_col in data.columns:
            # Extract the target column from DataFrame
            series_data = data[target_col]
        elif data.shape[1] > 0:
            # If target_col not found, use first column
            series_data = data.iloc[:, 0]
            if target_col is not None:
                print(f"Target column '{target_col}' not found. Using first column.")
        else:
            # Empty DataFrame, return empty Series
            return pd.Series()
    else:
        # Already a Series
        series_data = data
        
    # Calculate cumulative sum
    return series_data.cumsum()

def safe_prepare_cumulative_forecast(forecasts: Dict[str, Dict[str, Any]],
                                     historical_data: Union[pd.DataFrame, pd.Series],
                                     target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely prepare cumulative forecast results from multiple models.
    
    Args:
        forecasts: Dictionary of forecast model results
        historical_data: Historical data as DataFrame or Series
        target_col: Column name if historical_data is a DataFrame
        
    Returns:
        Dictionary with cumulative forecast results
    """
    try:
        # Get historical data as cumulative
        cum_historical = safely_get_cumulative_data(historical_data, target_col)
        
        result = {
            "historical": cum_historical,
            "models": {}
        }
        
        # Process each forecast model
        for model_name, forecast_dict in forecasts.items():
            if 'forecast' not in forecast_dict or forecast_dict['forecast'] is None:
                continue
                
            forecast = forecast_dict['forecast']
            
            # Get last historical value for continuity
            last_cum_value = cum_historical.iloc[-1] if not cum_historical.empty else 0
            
            # Calculate cumulative forecast
            cum_forecast = forecast.cumsum() + last_cum_value
            
            # Prepare result entry
            model_result = {
                'forecast': cum_forecast
            }
            
            # Add confidence intervals if available
            if 'lower_bound' in forecast_dict and 'upper_bound' in forecast_dict:
                lower_bound = forecast_dict['lower_bound']
                upper_bound = forecast_dict['upper_bound']
                
                if lower_bound is not None and upper_bound is not None:
                    model_result['lower_bound'] = lower_bound.cumsum() + last_cum_value
                    model_result['upper_bound'] = upper_bound.cumsum() + last_cum_value
            
            # Add to results
            result['models'][model_name] = model_result
            
        return result
        
    except Exception as e:
        print(f"Error preparing cumulative forecasts: {str(e)}")
        return {}
