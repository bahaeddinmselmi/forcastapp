"""
Helper module for month name detection and conversion
"""

import pandas as pd
from datetime import datetime

def detect_month_names(df, columns=None):
    """
    Detect columns containing month names in a dataframe
    
    Args:
        df: DataFrame to analyze
        columns: Specific columns to check (if None, check all columns)
        
    Returns:
        List of column names that likely contain month names
    """
    month_patterns = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    month_cols = []
    
    if columns is None:
        columns = df.columns
        
    for col in columns:
        if df[col].dtype == 'object':  # String data
            # Get first 10 non-null values
            sample = df[col].dropna().astype(str).iloc[:10].str.lower().tolist()
            
            # Check for month patterns
            if any(any(pattern in val for pattern in month_patterns) for val in sample):
                month_cols.append(col)
                
    return month_cols

# Import our enhanced date parser from the utility module
from utils.date_utils import parse_date_formats

def month_to_date(value, year=None):
    """
    Convert a month name to a date with specified year
    
    Args:
        value: String containing month name/abbreviation
        year: Year to use (defaults to current year)
        
    Returns:
        pandas Timestamp object or original value if conversion fails
    """
    # Use our enhanced parser
    parsed_date = parse_date_formats(value, selected_year=year, default_day=1)
    
    # Return the parsed date or original value if parsing failed
    if pd.isna(parsed_date):
        return value
    return parsed_date
