"""
Helper functions for market intelligence features.
Ensures robust handling of different data types and formats.
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Union, Dict, Any, Optional

def safely_get_best_forecast(forecasts: Dict[str, Dict[str, Any]], 
                           best_model: Optional[str] = None) -> pd.Series:
    """
    Safely get the best forecast from the forecasts dictionary.
    Falls back to the first available model if the best_model is not found.

    Args:
        forecasts: Dictionary of forecast models
        best_model: Name of the best model, if available

    Returns:
        The forecast data as a Series
    """
    # Return empty series if no forecasts
    if not forecasts:
        return pd.Series()
    
    # If best model specified and available, use it
    if best_model and best_model in forecasts:
        if 'forecast' in forecasts[best_model] and forecasts[best_model]['forecast'] is not None:
            return forecasts[best_model]['forecast'].copy()
    
    # Otherwise, find the first available forecast
    for model_name, model_data in forecasts.items():
        if 'forecast' in model_data and model_data['forecast'] is not None:
            print(f"Using {model_name} as fallback since best model was not available")
            return model_data['forecast'].copy()
    
    # No valid forecast found
    return pd.Series()

def ensure_datetime_index(series: pd.Series) -> pd.Series:
    """
    Ensures the Series has a DatetimeIndex, converting if needed.
    
    Args:
        series: The input Series
        
    Returns:
        Series with DatetimeIndex
    """
    # Already a DatetimeIndex, return as-is
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    
    # Create a new Series with proper DatetimeIndex
    try:
        # If index can be parsed as dates, convert directly
        new_index = pd.DatetimeIndex(pd.to_datetime(series.index))
        new_series = pd.Series(series.values, index=new_index)
        return new_series
    except:
        # Create synthetic dates starting from today
        start_date = pd.Timestamp.now()
        freq = 'MS'  # Monthly by default
        new_index = pd.date_range(start=start_date, periods=len(series), freq=freq)
        new_series = pd.Series(series.values, index=new_index)
        return new_series
        
def apply_market_intelligence(forecast: pd.Series,
                             promo_start: datetime.date,
                             promo_end: datetime.date,
                             promo_impact: float,
                             competitor_impact: float,
                             gdp_growth: float,
                             inflation: float) -> pd.Series:
    """
    Apply market intelligence adjustments to a forecast Series.
    
    Args:
        forecast: The forecast Series to adjust
        promo_start: Start date of promotion
        promo_end: End date of promotion
        promo_impact: Promotion impact percentage
        competitor_impact: Competitor impact percentage
        gdp_growth: GDP growth percentage
        inflation: Inflation rate percentage
        
    Returns:
        Adjusted forecast Series
    """
    # Make a copy to avoid modifying the original
    adjusted_forecast = forecast.copy()
    
    # Ensure datetime index
    adjusted_forecast = ensure_datetime_index(adjusted_forecast)
    
    # Apply adjustments
    for i, timestamp in enumerate(adjusted_forecast.index):
        try:
            # Convert timestamp to date for comparison
            forecast_date = timestamp.date()
            
            # Apply promotion impact
            if promo_start <= forecast_date <= promo_end:
                adjusted_forecast.iloc[i] *= (1 + promo_impact/100)
            
            # Apply competitor impact (first 3 months)
            if i < 3:
                adjusted_forecast.iloc[i] *= (1 + competitor_impact/100)
            
            # Apply economic factors
            adjusted_forecast.iloc[i] *= (1 + (gdp_growth - inflation)/100)
            
        except Exception as e:
            # Skip this point but continue with others
            print(f"Error adjusting forecast point at {timestamp}: {str(e)}")
            continue
    
    return adjusted_forecast
