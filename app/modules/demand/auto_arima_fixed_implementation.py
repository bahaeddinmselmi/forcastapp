"""
This file contains a completely rewritten Auto ARIMA implementation to replace the
problematic section in the UI.py file. This new implementation has clean indentation
and ensures proper forecast generation.
"""

import pandas as pd
import numpy as np
import streamlit as st
from modules.demand.forecast_values import generate_trend_fallback_forecast

def run_auto_arima_model(train_data, target_col, forecast_periods, future_index):
    """
    Clean implementation of Auto ARIMA model logic for the demand planning UI
    
    Args:
        train_data: Training DataFrame with the target column
        target_col: Target column name to forecast
        forecast_periods: Number of periods to forecast
        future_index: DateTimeIndex for forecast values
        
    Returns:
        Dictionary with auto_arima_result and success status
    """
    try:
        # Import our clean Auto ARIMA implementation
        from modules.demand.auto_arima_mod import generate_auto_arima_forecast
        
        # Check if target_col exists in the data
        if isinstance(train_data, pd.DataFrame):
            if target_col not in train_data.columns:
                # If target_col doesn't exist but we have a Series or single column DataFrame
                if train_data.shape[1] == 1:
                    # Use the first column
                    actual_target = train_data.columns[0]
                    st.info(f"Target column '{target_col}' not found. Using '{actual_target}' instead.")
                    target_data = train_data[actual_target]
                else:
                    raise KeyError(f"Target column '{target_col}' not found in data and multiple columns exist.")
            else:
                target_data = train_data[target_col]
        else:
            # Handle case where train_data is already a Series
            st.info(f"Using provided Series data for forecasting.")
            target_data = train_data
        
        with st.spinner("Running Auto ARIMA model (this may take a moment)..."):
            # Generate forecast using our clean implementation
            auto_arima_result = generate_auto_arima_forecast(
                train_data=target_data,
                periods=forecast_periods,
                future_index=future_index
            )
            
            # Display success message
            st.success(f"Auto ARIMA forecast generated successfully.")
            
            # Display a sample of the forecast values for validation
            forecast_sample = auto_arima_result['forecast'].head(3).to_dict()
            st.info(f"Sample Auto ARIMA forecast values: {forecast_sample}")
            
            # Return the successful result
            return {
                'result': auto_arima_result,
                'success': True
            }
    except Exception as e:
        # Handle any errors
        st.error(f"Error in Auto ARIMA model: {e}")
        
        # Create a basic fallback forecast
        if future_index is not None:
            # Use same target_data handling logic as above
            if isinstance(train_data, pd.DataFrame):
                if target_col not in train_data.columns:
                    if train_data.shape[1] == 1:
                        actual_target = train_data.columns[0]
                        fallback_data = train_data[actual_target]
                    else:
                        # If we have multiple columns but none match target_col, use the first numeric column
                        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            fallback_data = train_data[numeric_cols[0]]
                            st.info(f"Using '{numeric_cols[0]}' for fallback forecast.")
                        else:
                            raise KeyError(f"No numeric columns found for fallback forecast.")
                else:
                    fallback_data = train_data[target_col]
            else:
                # Handle Series data
                fallback_data = train_data
                
            trend_forecast = generate_trend_fallback_forecast(
                fallback_data, 
                forecast_periods, 
                future_index
            )
            
            # Return the fallback result
            return {
                'result': {
                    'forecast': trend_forecast,
                    'lower_bound': trend_forecast * 0.9,
                    'upper_bound': trend_forecast * 1.1,
                    'model': 'ARIMA Fallback',
                    'error': str(e)
                },
                'success': False,
                'error': str(e)
            }
        else:
            # No future index available
            return {
                'result': None,
                'success': False,
                'error': str(e)
            }
