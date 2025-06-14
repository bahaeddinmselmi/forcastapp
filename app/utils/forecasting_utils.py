"""
Additional Forecasting utilities
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional

def parse_month_abbreviation(date_str):
    """
    Parse a month abbreviation like 'Mar-20' to a pandas datetime object
    
    Args:
        date_str: A string representing a month like 'Mar-20' or 'Jan-21'
        
    Returns:
        pd.Timestamp: A pandas Timestamp object
    """
    try:
        # Handle formats like 'Mar-20' (Month-Year)
        if isinstance(date_str, str) and len(date_str) <= 7 and '-' in date_str:
            parts = date_str.split('-')
            if len(parts) == 2:
                month_abbr = parts[0].strip()
                year_str = parts[1].strip()
                
                # Convert month abbreviation to number
                from calendar import month_abbr
                month_names = list(month_abbr)
                # Case insensitive search
                month_idx = next((i for i, m in enumerate(month_names) 
                                 if m.lower() == month_abbr.lower()), None)
                
                if month_idx is None:
                    # Try to match partial month names
                    month_idx = next((i for i, m in enumerate(month_names) 
                                    if m.lower().startswith(month_abbr.lower())), None)
                
                if month_idx is not None:
                    # Add '20' prefix if the year is just 2 digits
                    if len(year_str) == 2:
                        year_str = '20' + year_str
                    
                    # Create a date string in a format pandas can parse
                    date_str = f"{year_str}-{month_idx:02d}-01"
                    return pd.to_datetime(date_str)
        
        # If we couldn't parse it with our custom logic, try pandas directly
        return pd.to_datetime(date_str)
    except Exception as e:
        raise ValueError(f"Could not parse date: {date_str}. Error: {str(e)}")

def safely_parse_date(date_value):
    """
    Safely parse a date value, handling various formats including month abbreviations.
    
    Args:
        date_value: The date value to parse (can be string, timestamp, etc.)
    
    Returns:
        pd.Timestamp or None if parsing fails
    """
    if pd.isna(date_value):
        return None
        
    if isinstance(date_value, pd.Timestamp):
        return date_value
        
    try:
        # Try special handling for month abbreviations
        if isinstance(date_value, str) and len(date_value) <= 7 and '-' in date_value:
            return parse_month_abbreviation(date_value)
        
        # Try standard pandas parsing
        return pd.to_datetime(date_value)
    except Exception as e:
        print(f"Warning: Could not parse date '{date_value}': {str(e)}")
        return None

def create_future_date_range(last_date, periods, freq='MS'):
    """
    Create a date range for forecasting, starting from the month after the last date in dataset
    
    Args:
        last_date: The last date in the dataset
        periods: Number of periods to forecast
        freq: Frequency of dates ('MS' for month start, 'D' for daily, etc)
        
    Returns:
        DatetimeIndex: A pandas DatetimeIndex with future dates
    """
    # Try to handle different date formats
    try:
        # Convert to datetime if it's not already
        if not isinstance(last_date, pd.Timestamp):
            last_date = safely_parse_date(last_date)
            
        # If parsing failed, use current date as fallback
        if last_date is None:
            import datetime
            last_date = datetime.datetime.now()
            
        # Start the forecast from the last date
        start_date = last_date
            
        # Create the date range
        future_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        return future_dates
    except Exception as e:
        # If all else fails, create a generic date range starting from today
        import datetime
        today = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = today + pd.DateOffset(months=1)
        return pd.date_range(start=next_month, periods=periods, freq=freq)

def get_data_frequency(date_index):
    """
    Determine the frequency of a date index
    
    Args:
        date_index: A pandas DatetimeIndex
        
    Returns:
        str: The frequency string ('MS', 'D', etc.) or 'MS' as default
    """
    if not isinstance(date_index, pd.DatetimeIndex):
        return 'MS'  # Default to monthly if not a datetime index
        
    try:
        # Try pandas built-in frequency detection
        inferred_freq = pd.infer_freq(date_index)
        if inferred_freq is not None:
            return inferred_freq
            
        # If pandas can't detect, try to determine based on average days difference
        if len(date_index) > 1:
            # Sort the index to ensure correct differences
            sorted_index = date_index.sort_values()
            # Calculate average days between dates
            diffs = sorted_index[1:] - sorted_index[:-1]
            avg_days = diffs.mean().days
            
            if avg_days >= 28 and avg_days <= 31:
                return 'MS'  # Monthly
            elif avg_days >= 90 and avg_days <= 92:
                return 'QS'  # Quarterly
            elif avg_days >= 365 and avg_days <= 366:
                return 'AS'  # Annual
            elif avg_days >= 6 and avg_days <= 8:
                return 'W'   # Weekly
            elif avg_days >= 0.9 and avg_days <= 1.1:
                return 'D'   # Daily
                
    except Exception:
        pass
        
    # Default to monthly if all else fails
    return 'MS'
