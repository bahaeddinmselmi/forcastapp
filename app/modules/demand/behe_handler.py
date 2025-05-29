"""
Specialized handler for MMM-YY dates and duplicate date handling
This module provides clean helper functions for the IBP application
"""

import pandas as pd
import streamlit as st
import re
from datetime import datetime

def add_year_toggle():
    """
    Adds a UI control for toggling between forced year and original years
    Returns the selected year and whether to force it
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

def handle_duplicate_dates(df, date_col):
    """
    Provides UI options for handling duplicate dates in a dataframe
    
    Args:
        df: DataFrame containing the data
        date_col: Name of the column containing dates
        
    Returns:
        DataFrame with duplicates handled according to user selection
    """
    # Check for duplicates
    duplicates = df[date_col].duplicated().sum()
    
    # If no duplicates, return original
    if duplicates == 0:
        return df
        
    # Show warning and options
    st.warning(f"Duplicate dates detected in your data ({duplicates} duplicates found).")
    
    # Radio buttons for duplicate handling options
    option = st.radio(
        "How would you like to handle duplicate dates?",
        options=[
            "Sum values for duplicate dates", 
            "Average values for duplicate dates", 
            "Keep first occurrence only", 
            "Keep last occurrence only"
        ],
        index=0  # Default to sum
    )
    
    # Apply selected method
    if option == "Sum values for duplicate dates":
        st.info("Summing values for duplicate dates")
        result = df.groupby(date_col).sum().reset_index()
    elif option == "Average values for duplicate dates":
        st.info("Averaging values for duplicate dates")
        result = df.groupby(date_col).mean().reset_index()
    elif option == "Keep first occurrence only":
        st.info("Keeping first occurrence of each date")
        result = df.drop_duplicates(subset=[date_col], keep='first')
    elif option == "Keep last occurrence only":
        st.info("Keeping last occurrence of each date")
        result = df.drop_duplicates(subset=[date_col], keep='last')
    else:
        # Fallback to summing if something goes wrong
        result = df.groupby(date_col).sum().reset_index()
    
    # Show summary
    st.success(f"After handling duplicates: {len(result)} data points remaining")
    
    return result

def detect_behe_format(df):
    """
    Check if the dataframe matches the behe.xlsx format pattern
    """
    # Check for specific markers of behe.xlsx format
    if 'months' in df.columns or any('ASPEGIC' in str(col) for col in df.columns):
        return True
    
    # Check first row for month header
    if 'months' in df.columns and isinstance(df['months'].iloc[0], str) and df['months'].iloc[0].lower() == 'months':
        return True
    
    # Look for MMM-YY pattern in any column
    for col in df.columns:
        if df[col].dtype == 'object':  # String data
            sample = df[col].dropna().astype(str).head(5).tolist()
            # Check for MMM-YY pattern (e.g., "Mar-20", "Apr-20")
            if any(re.match(r'^[A-Za-z]{3}-\d{2}$', str(val)) for val in sample):
                return True
    
    return False

def clean_behe_dataframe(df):
    """
    Clean and restructure the behe.xlsx dataframe for proper processing
    """
    st.info("Detected behe.xlsx format. Applying specialized handling.")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Check if first row contains column headers
    first_row_is_header = False
    if 'months' in df_clean.columns and isinstance(df_clean['months'].iloc[0], str) and df_clean['months'].iloc[0].lower() == 'months':
        first_row_is_header = True
        new_cols = df_clean.iloc[0].values
        df_clean = df_clean.iloc[1:].copy()
        df_clean.columns = new_cols
        st.success("Fixed column structure. First row contained column headers.")
    
    # Find the date column
    date_col = None
    for col in df_clean.columns:
        if col.lower() == 'months' or col.lower() == 'month' or col.lower() == 'date':
            date_col = col
            break
    
    if not date_col:
        # Try to find a column with MMM-YY format
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # String data
                sample = df_clean[col].dropna().astype(str).head(5).tolist()
                if any(re.match(r'^[A-Za-z]{3}-\d{2}$', str(val)) for val in sample):
                    date_col = col
                    break
    
    if not date_col:
        st.warning("Could not identify date column in behe format.")
        return df_clean, None, None
    
    # Find a suitable value column - preferring numeric columns
    value_cols = []
    for col in df_clean.columns:
        if col != date_col:
            try:
                # Check if column has numeric values
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if not df_clean[col].isna().all():
                    value_cols.append(col)
            except:
                continue
    
    # Use the first value column or let user select
    value_col = None
    if len(value_cols) == 1:
        value_col = value_cols[0]
    elif len(value_cols) > 1:
        # Sort by non-null count descending (prefer columns with more data)
        value_cols_sorted = sorted(value_cols, key=lambda x: df_clean[x].count(), reverse=True)
        value_col = st.selectbox("Select the value column to use for forecasting:", value_cols_sorted)
    
    # Parse dates with proper handling for MMM-YY format
    if date_col:
        from utils.date_utils import parse_date_formats
        
        # Get the selected year if available
        if 'config' in st.session_state and 'selected_year' in st.session_state['config']:
            selected_year = st.session_state['config'].get('selected_year')
        else:
            selected_year = None
            
        # Apply our enhanced date parser to handle the MMM-YY format
        df_clean[date_col] = df_clean[date_col].apply(
            lambda x: parse_date_formats(x, selected_year=selected_year)
        )
        
        # Handle duplicate dates with a user selection
        duplicates = df_clean[date_col].duplicated().sum()
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate dates in your behe.xlsx file.")
            
            # Get user preference for handling duplicates
            dup_method = st.selectbox(
                "How would you like to handle duplicate dates?",
                options=[
                    "Sum values for duplicate dates",
                    "Average values for duplicate dates",
                    "Keep most recent data for each date",
                    "Keep earliest data for each date"
                ],
                index=0  # Default to sum
            )
            
            # Apply the selected method
            if dup_method == "Sum values for duplicate dates":
                df_clean = df_clean.groupby(date_col).sum().reset_index()
                st.success("Summed values for duplicate dates")
            elif dup_method == "Average values for duplicate dates":
                df_clean = df_clean.groupby(date_col).mean().reset_index()
                st.success("Averaged values for duplicate dates")
            elif dup_method == "Keep most recent data for each date":
                df_clean = df_clean.drop_duplicates(subset=[date_col], keep='last')
                st.success("Kept most recent data for each date")
            elif dup_method == "Keep earliest data for each date":
                df_clean = df_clean.drop_duplicates(subset=[date_col], keep='first')
                st.success("Kept earliest data for each date")
    
    return df_clean, date_col, value_col

def prepare_behe_timeseries(df, date_col, value_col):
    """
    Convert cleaned behe dataframe into a proper time series
    """
    # Make sure we have valid columns
    if date_col is None or value_col is None:
        st.error("Missing valid date or value column for time series creation.")
        return None
    
    try:
        # Create a copy to avoid modifying the original
        ts_df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(ts_df[date_col]):
            from utils.date_utils import parse_date_formats
            ts_df[date_col] = ts_df[date_col].apply(
                lambda x: parse_date_formats(x, selected_year=None)
            )
            
        # Drop rows with invalid dates
        ts_df = ts_df.dropna(subset=[date_col])
        
        # Create time series with date as index
        ts_df = ts_df.set_index(date_col)
        
        # Only keep the value column
        if value_col in ts_df.columns:
            ts_df = ts_df[[value_col]]
        
        # Sort by date
        ts_df = ts_df.sort_index()
        
        # Ensure monthly frequency
        if ts_df.index.freq is None:
            # Create a proper monthly date range
            date_range = pd.date_range(start=ts_df.index.min(), end=ts_df.index.max(), freq='MS')
            
            # Reindex the dataframe with the new date range
            ts_df = ts_df.reindex(date_range)
            
            # Fill missing values with interpolation
            ts_df = ts_df.interpolate(method='linear')
        
        return ts_df
    
    except Exception as e:
        st.error(f"Error preparing time series: {str(e)}")
        return None
