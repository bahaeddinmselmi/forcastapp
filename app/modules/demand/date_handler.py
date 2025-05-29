"""
Specialized date handling module for IBP application
This helps with handling special date formats like MMM-YY and duplicate date issues
"""

import pandas as pd
import streamlit as st
import re
from datetime import datetime
import numpy as np

def handle_duplicate_dates(df, date_col):
    """
    Interactive handling of duplicate dates with user options
    
    Args:
        df: DataFrame with date column
        date_col: Column name containing date values
        
    Returns:
        DataFrame with duplicate dates handled according to user choice
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check for duplicates
    duplicates = df_copy[date_col].duplicated().sum()
    
    if duplicates > 0:
        st.warning(f"Duplicate dates detected in your data ({duplicates} duplicates found)")
        
        # Add options for duplicate handling
        dup_option = st.selectbox(
            "How would you like to handle duplicate dates?",
            options=[
                "Sum values for duplicate dates", 
                "Average values for duplicate dates", 
                "Keep first occurrence only", 
                "Keep last occurrence only"
            ],
            index=0
        )
        
        # Apply the chosen method
        if dup_option == "Sum values for duplicate dates":
            st.info("Summing values for duplicate dates")
            df_copy = df_copy.groupby(date_col).sum().reset_index()
        elif dup_option == "Average values for duplicate dates":
            st.info("Averaging values for duplicate dates")
            df_copy = df_copy.groupby(date_col).mean().reset_index()
        elif dup_option == "Keep first occurrence only":
            st.info("Keeping first occurrence of each date")
            df_copy = df_copy.drop_duplicates(subset=[date_col], keep='first')
        elif dup_option == "Keep last occurrence only":
            st.info("Keeping last occurrence of each date")
            df_copy = df_copy.drop_duplicates(subset=[date_col], keep='last')
    
    return df_copy

def year_setting_control():
    """
    Provides a UI control for selecting a year and toggling whether to force it
    
    Returns:
        tuple: (selected_year, force_year)
    """
    # Current year for default selection
    current_year = datetime.now().year
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    # Year selection in first column
    with col1:
        selected_year = st.selectbox(
            "Select Year for Month Data:",
            options=list(range(current_year - 10, current_year + 2)),
            index=10  # Default to current year
        )
    
    # Toggle for forcing year in second column
    with col2:
        force_year = st.checkbox(
            "Force this year", 
            value=False,
            help="If checked, will override original years in your data (like 'Mar-20') with the selected year"
        )
    
    # Store in session state
    if 'config' not in st.session_state:
        st.session_state['config'] = {}
    st.session_state['config']['selected_year'] = selected_year
    st.session_state['force_selected_year'] = force_year
    
    # Show explanation
    if force_year:
        st.warning(f"All dates will use {selected_year} as the year, ignoring any year information in your data")
    else:
        st.success(f"Original years in your data (like in 'Mar-20') will be preserved. {selected_year} will only be used for dates without year information.")
    
    return selected_year, force_year
    
    # Year selection
    with col1:
        selected_year = st.selectbox(
            "Select Year for Month Data:",
            options=list(range(current_year - 10, current_year + 2)),
            index=10  # Default to current year
        )
    
    # Force year toggle
    with col2:
        force_year = st.checkbox(
            "Force selected year", 
            value=False,
            help="When enabled, overrides years in formats like 'Mar-20' with the selected year"
        )
    
    # Store settings in session state
    if 'config' not in st.session_state:
        st.session_state['config'] = {}
    st.session_state['config']['selected_year'] = selected_year
    st.session_state['force_selected_year'] = force_year
    
    # Show explanation based on setting
    if force_year:
        st.warning(f"All dates will use {selected_year} as the year, ignoring any year information in your data")
    else:
        st.success(f"Years in your data (like in 'Mar-20') will be preserved. {selected_year} will only be used for dates without year information")
    
    return selected_year, force_year

def parse_mmm_yy_date(value, selected_year=None, force_year=False):
    """
    Special parser for MMM-YY format like 'Mar-20'
    
    Args:
        value: String to parse
        selected_year: Year to use if forcing or if no year in value
        force_year: Whether to force the selected year
        
    Returns:
        Pandas timestamp or original value
    """
    if pd.isna(value):
        return value
        
    # Handle non-string values
    if not isinstance(value, str):
        try:
            value = str(value)
        except:
            return value
    
    # Clean up the string
    value = value.strip()
    
    # Check for MMM-YY pattern (Mar-20)
    mmm_yy_pattern = re.compile(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\-\s]?(\d{2})$', re.IGNORECASE)
    match = mmm_yy_pattern.match(value.lower())
    
    if match:
        month_name = match.group(1).lower()
        year_digits = match.group(2)
        
        # Month name to number mapping
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        month_num = month_map.get(month_name)
        if month_num:
            # Convert 2-digit year to 4-digit
            if int(year_digits) > 50:  # Assume 19xx for years > 50
                year = 1900 + int(year_digits)
            else:  # Assume 20xx for years <= 50
                year = 2000 + int(year_digits)
            
            # Use selected year if forcing years
            if force_year and selected_year is not None:
                year = selected_year
                
            # Create timestamp with the first day of the month
            try:
                return pd.Timestamp(year=year, month=month_num, day=1)
            except:
                pass
    
    # If not MMM-YY or parsing failed, return original
    return value
