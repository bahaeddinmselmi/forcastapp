"""
Enhanced Prophet Forecasting Model

This module provides an improved implementation of Facebook Prophet with 
automatic seasonality detection, holiday effects, and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Dict, Any, Optional, List, Tuple
import holidays
import logging
from datetime import datetime, timedelta

def enhanced_prophet_forecast(
    train_data: pd.DataFrame,
    periods: int,
    date_col: str = 'ds',
    target_col: str = 'y',
    return_components: bool = False,
    future_df: Optional[pd.DataFrame] = None,
    future_index: Optional[pd.DatetimeIndex] = None,
    country_code: str = 'US',
    auto_seasonality: bool = True,
    tune_parameters: bool = True) -> Dict[str, Any]:
    """
    Enhanced forecast using Facebook Prophet with advanced features.
    
    Args:
        train_data: Training data as pandas DataFrame with date and target columns
        periods: Number of periods to forecast
        date_col: Name of the date column
        target_col: Name of the target column
        return_components: Whether to return trend, seasonality components
        future_df: Optional custom future dataframe for forecast
        future_index: Optional custom index for future dates
        country_code: Country code for holidays (default: US)
        auto_seasonality: Whether to automatically detect and apply seasonality
        tune_parameters: Whether to tune hyperparameters
        
    Returns:
        Dictionary with forecast results
    """
    try:
        # Copy data to avoid modifying original
        df = train_data.copy()
        
        # Ensure data is sorted
        df = df.sort_values(by=date_col)
        
        # Feature engineering: add derived date features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        
        # Analyze the data frequency
        freq, is_daily, is_weekly, is_monthly, is_quarterly, is_yearly = detect_frequency(df[date_col])
        
        # Detect seasonality patterns
        seasonality_params = determine_seasonality(df, freq, auto_seasonality)
        
        # Create holidays dataframe if country provided
        holiday_df = None
        if country_code:
            holiday_df = create_holiday_df(train_data[date_col].min(), 
                                          future_index.max() if future_index is not None else None,
                                          periods, country_code)
        
        # Create Prophet model with optimized parameters
        model = create_prophet_model(seasonality_params, holiday_df, tune_parameters, df)
        
        # Fit the model
        model.fit(df)
        
        # Generate future dataframe
        future = create_future_df(model, df, periods, future_df, future_index, freq)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Prepare results
        result = {
            'forecast': pd.Series(forecast['yhat'].values, index=forecast['ds']),
            'lower_bound': pd.Series(forecast['yhat_lower'].values, index=forecast['ds']),
            'upper_bound': pd.Series(forecast['yhat_upper'].values, index=forecast['ds']),
            'model': model,
            'forecast_df': forecast
        }
        
        # Add components if requested
        if return_components:
            result['components'] = {
                'trend': pd.Series(forecast['trend'].values, index=forecast['ds'])
            }
            
            # Add seasonality components if available
            if 'yearly' in seasonality_params and seasonality_params['yearly']:
                result['components']['yearly'] = pd.Series(forecast['yearly'].values, index=forecast['ds'])
                
            if 'weekly' in seasonality_params and seasonality_params['weekly']:
                result['components']['weekly'] = pd.Series(forecast['weekly'].values, index=forecast['ds'])
                
            if 'monthly' in seasonality_params and seasonality_params['monthly']:
                result['components']['monthly'] = pd.Series(forecast['monthly'].values, index=forecast['ds'])
                
            if 'quarterly' in seasonality_params and seasonality_params['quarterly']:
                result['components']['quarterly'] = pd.Series(forecast['quarterly'].values, index=forecast['ds'])
                
            if 'daily' in seasonality_params and seasonality_params['daily']:
                result['components']['daily'] = pd.Series(forecast['daily'].values, index=forecast['ds'])
        
        return result
    
    except Exception as e:
        logging.error(f"Error in enhanced Prophet forecast: {str(e)}")
        raise
        
def detect_frequency(date_series: pd.Series) -> Tuple[str, bool, bool, bool, bool, bool]:
    """
    Detect the frequency of the time series data.
    
    Args:
        date_series: Series containing dates
        
    Returns:
        Tuple of (frequency string, is_daily, is_weekly, is_monthly, is_quarterly, is_yearly)
    """
    if len(date_series) < 2:
        return 'MS', False, False, True, False, False  # Default to monthly
    
    # Calculate differences between consecutive dates
    sorted_dates = sorted(date_series)
    date_diffs = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
    
    if not date_diffs:
        return 'MS', False, False, True, False, False  # Default to monthly
    
    avg_diff = sum(date_diffs) / len(date_diffs)
    
    # Determine frequency based on average difference
    is_daily = avg_diff < 2
    is_weekly = 6 <= avg_diff <= 8
    is_monthly = 28 <= avg_diff <= 31
    is_quarterly = 89 <= avg_diff <= 92
    is_yearly = 350 <= avg_diff <= 370
    
    if is_daily:
        return 'D', True, False, False, False, False
    elif is_weekly:
        return 'W', False, True, False, False, False
    elif is_monthly:
        return 'MS', False, False, True, False, False
    elif is_quarterly:
        return 'QS', False, False, False, True, False
    elif is_yearly:
        return 'AS', False, False, False, False, True
    else:
        # Default to monthly if can't determine
        return 'MS', False, False, True, False, False

def determine_seasonality(df: pd.DataFrame, freq: str, auto_seasonality: bool) -> Dict[str, bool]:
    """
    Determine appropriate seasonality parameters based on data frequency.
    
    Args:
        df: Training data
        freq: Detected frequency
        auto_seasonality: Whether to automatically detect seasonality
        
    Returns:
        Dictionary of seasonality parameters
    """
    # Default seasonality settings
    seasonality = {
        'daily': False,
        'weekly': False,
        'monthly': False,
        'quarterly': False,
        'yearly': False
    }
    
    if not auto_seasonality:
        # Basic seasonality based on frequency
        if freq == 'D':
            seasonality['daily'] = True
            seasonality['weekly'] = True
        elif freq == 'W':
            seasonality['weekly'] = True
            seasonality['yearly'] = True
        elif freq == 'MS':
            seasonality['monthly'] = True
            seasonality['yearly'] = True
        elif freq == 'QS':
            seasonality['quarterly'] = True
            seasonality['yearly'] = True
        elif freq == 'AS':
            seasonality['yearly'] = True
        return seasonality
    
    # Auto-detect seasonality
    # Check if we have enough data for yearly seasonality (at least 2 years)
    date_range = df['ds'].max() - df['ds'].min()
    if date_range.days >= 730:  # Roughly 2 years
        seasonality['yearly'] = True
    
    # Check for weekly seasonality (need at least 2 weeks of daily data)
    if freq == 'D' and date_range.days >= 14:
        seasonality['weekly'] = True
    
    # Check for monthly seasonality (need several months of at least weekly data)
    if (freq == 'D' or freq == 'W') and date_range.days >= 90:
        seasonality['monthly'] = True
    
    # Daily seasonality for sub-daily data (not typically present in this application)
    if freq == 'D' and len(df) > 2:
        # Check times per day to see if we have intraday data
        dates_count = df['ds'].dt.date.value_counts()
        if dates_count.max() > 3:  # If we have more than 3 observations per day
            seasonality['daily'] = True
    
    # Quarterly is mainly for yearly data with quarterly observations
    if (freq == 'MS' or freq == 'QS') and date_range.days >= 450:
        seasonality['quarterly'] = True
        
    return seasonality

def create_holiday_df(start_date: datetime, 
                     end_date: Optional[datetime], 
                     periods: int, 
                     country_code: str) -> pd.DataFrame:
    """
    Create a holiday dataframe for Prophet.
    
    Args:
        start_date: Start date of training data
        end_date: End date of future data (if available)
        periods: Number of periods to forecast
        country_code: Country code for holidays
        
    Returns:
        DataFrame with holiday information
    """
    try:
        # If end_date not provided, estimate it based on periods
        if end_date is None:
            end_date = start_date + timedelta(days=periods * 30)  # Rough estimate
            
        # Get country holidays
        country_holidays = holidays.country_holidays(country_code)
        
        # Create date range
        all_dates = pd.date_range(start=start_date, end=end_date)
        
        # Find holidays in date range
        holiday_dates = []
        holiday_names = []
        
        for date in all_dates:
            if date in country_holidays:
                holiday_dates.append(date)
                holiday_names.append(country_holidays.get(date))
        
        # Create holiday dataframe
        holiday_df = pd.DataFrame({
            'ds': holiday_dates,
            'holiday': holiday_names
        })
        
        return holiday_df
    except:
        # If any error occurs (like invalid country code), return None
        return None

def create_prophet_model(seasonality: Dict[str, bool], 
                        holiday_df: Optional[pd.DataFrame],
                        tune_parameters: bool,
                        train_data: pd.DataFrame) -> Prophet:
    """
    Create a Prophet model with optimized parameters.
    
    Args:
        seasonality: Dictionary of seasonality parameters
        holiday_df: DataFrame with holiday information
        tune_parameters: Whether to tune hyperparameters
        train_data: Training data
        
    Returns:
        Configured Prophet model
    """
    # Start with base parameters
    params = {
        'yearly_seasonality': seasonality.get('yearly', False),
        'weekly_seasonality': seasonality.get('weekly', False),
        'daily_seasonality': seasonality.get('daily', False),
    }
    
    # Use holidays if available
    if holiday_df is not None and len(holiday_df) > 0:
        params['holidays'] = holiday_df
    
    if tune_parameters and len(train_data) >= 30:
        # For smaller datasets, use more conservative parameters
        if len(train_data) < 100:
            params.update({
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'changepoint_range': 0.8,
                'interval_width': 0.9,
            })
        # For medium datasets, use moderate parameters
        elif len(train_data) < 500:
            params.update({
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'changepoint_range': 0.8,
                'interval_width': 0.95,
            })
        # For larger datasets, use more flexible parameters
        else:
            params.update({
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 15.0,
                'changepoint_range': 0.9,
                'interval_width': 0.95,
            })
    else:
        # Default parameters if not tuning
        params.update({
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'changepoint_range': 0.8,
            'interval_width': 0.9,
        })
    
    # Create the model
    model = Prophet(**params)
    
    # Add monthly seasonality if specified
    if seasonality.get('monthly', False):
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add quarterly seasonality if specified
    if seasonality.get('quarterly', False):
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    
    return model

def create_future_df(model: Prophet,
                   df: pd.DataFrame,
                   periods: int,
                   future_df: Optional[pd.DataFrame] = None,
                   future_index: Optional[pd.DatetimeIndex] = None,
                   freq: str = 'MS') -> pd.DataFrame:
    """
    Create future dataframe for prediction.
    
    Args:
        model: Fitted Prophet model
        df: Training data
        periods: Number of periods to forecast
        future_df: Optional custom future dataframe
        future_index: Optional custom future index
        freq: Frequency of data
        
    Returns:
        Future dataframe for prediction
    """
    if future_df is not None:
        # Use provided future dataframe
        return future_df
    
    if future_index is not None:
        # Use provided future index
        return pd.DataFrame({'ds': future_index})
    
    # Get the last date from training data
    last_date = df['ds'].max()
    
    # Create future dates based on frequency
    if freq == 'MS':
        # Monthly - start from first day of next month
        next_period = last_date.replace(day=1) + pd.DateOffset(months=1)
        future_dates = pd.date_range(start=next_period, periods=periods, freq=freq)
    elif freq == 'QS':
        # Quarterly - start from first day of next quarter
        next_quarter_month = ((last_date.month - 1) // 3 + 1) * 3 + 1
        if next_quarter_month > 12:
            next_quarter_month = 1
            next_quarter_year = last_date.year + 1
        else:
            next_quarter_year = last_date.year
        next_period = datetime(next_quarter_year, next_quarter_month, 1)
        future_dates = pd.date_range(start=next_period, periods=periods, freq=freq)
    elif freq == 'W':
        # Weekly - start from next week
        next_period = last_date + pd.DateOffset(weeks=1)
        future_dates = pd.date_range(start=next_period, periods=periods, freq=freq)
    elif freq == 'D':
        # Daily - start from next day
        next_period = last_date + pd.DateOffset(days=1)
        future_dates = pd.date_range(start=next_period, periods=periods, freq=freq)
    elif freq == 'AS':
        # Yearly - start from first day of next year
        next_period = datetime(last_date.year + 1, 1, 1)
        future_dates = pd.date_range(start=next_period, periods=periods, freq=freq)
    else:
        # Default to model's make_future_dataframe
        return model.make_future_dataframe(periods=periods, freq=freq)
    
    # Create the dataframe
    future = pd.DataFrame({'ds': future_dates})
    
    return future
