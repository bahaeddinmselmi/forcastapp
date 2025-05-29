"""
Helper functions for forecast metrics calculation with robust handling of both
DataFrame and Series data types.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import streamlit as st

def safe_evaluate_forecasts(actuals: Union[pd.Series, pd.DataFrame], 
                            forecasts: Dict[str, Dict[str, Any]],
                            target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Safely evaluate multiple forecast models handling both Series and DataFrame inputs.
    
    Args:
        actuals: Series or DataFrame with actual values
        forecasts: Dictionary of model forecasts (model_name -> forecast_dict)
        target_col: Target column name (used if actuals is DataFrame)
        
    Returns:
        DataFrame with evaluation metrics for each model
    """
    # Handle actuals - convert to Series if it's a DataFrame with target_col
    if isinstance(actuals, pd.DataFrame):
        if target_col is not None and target_col in actuals.columns:
            actuals_series = actuals[target_col]
        elif not actuals.empty and actuals.shape[1] == 1:
            # If only one column, use it
            actuals_series = actuals.iloc[:, 0]
        else:
            st.warning(f"Could not extract target column from DataFrame. Available columns: {', '.join(actuals.columns)}")
            # Create a dummy series to avoid errors
            actuals_series = pd.Series()
    else:
        # Already a Series
        actuals_series = actuals
    
    results = []
    
    # Check if we have valid actuals
    have_actuals = actuals_series is not None and len(actuals_series) > 0
    
    # Process each model's forecast
    for model_name, forecast_dict in forecasts.items():
        if 'forecast' not in forecast_dict:
            # Skip invalid forecast dictionaries
            results.append({
                'Model': model_name,
                'RMSE': np.nan,
                'MAPE': np.nan,
                'Error': 'Invalid forecast format'
            })
            continue
        
        # Get the forecast series from the model result
        forecast = forecast_dict['forecast']
        
        if have_actuals:
            # Make sure we're working with aligned indices for comparison
            common_index = actuals_series.index.intersection(forecast.index)
            
            if len(common_index) > 0:
                # Align both series on the common index
                actuals_aligned = actuals_series.loc[common_index]
                forecast_aligned = forecast.loc[common_index]
                
                # Calculate metrics
                try:
                    # Calculate RMSE
                    squared_errors = (actuals_aligned - forecast_aligned) ** 2
                    rmse = np.sqrt(squared_errors.mean())
                    
                    # Calculate MAPE with protection against division by zero
                    abs_pct_errors = np.abs((actuals_aligned - forecast_aligned) / actuals_aligned.replace(0, np.nan)) * 100
                    mape = abs_pct_errors.mean()
                    
                    results.append({
                        'Model': model_name,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                except Exception as e:
                    # If metrics calculation fails, add NaN values
                    results.append({
                        'Model': model_name,
                        'RMSE': np.nan,
                        'MAPE': np.nan,
                        'Error': str(e)
                    })
            else:
                # No common index points
                results.append({
                    'Model': model_name,
                    'RMSE': np.nan,
                    'MAPE': np.nan,
                    'Error': 'No overlapping time periods for evaluation'
                })
        else:
            # If we don't have actuals, return NaN metrics
            results.append({
                'Model': model_name,
                'RMSE': np.nan,
                'MAPE': np.nan,
                'Error': 'No actual data available for comparison'
            })
    
    # Create a DataFrame with results
    if results:
        return pd.DataFrame(results)
    else:
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=['Model', 'RMSE', 'MAPE', 'Error'])
