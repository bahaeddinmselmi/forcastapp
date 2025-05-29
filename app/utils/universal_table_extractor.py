"""
Universal table extractor that can handle various data formats.
Identifies and extracts time series data from any tabular format.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Dict, Any
import re

def extract_table_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Extract time series data from any table format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of:
        - DataFrame with date and value columns
        - Date column name
        - Value column name
    """
    # Show original data for reference
    st.subheader("Original Data")
    st.dataframe(df.head(10))
    
    # Step 1: Determine if the table has headers or if data starts immediately
    # Look for patterns in first few rows to determine which is the header
    header_row = detect_header_row(df)
    
    if header_row is not None:
        st.success(f"✅ Detected header row at index {header_row}")
        
        # Use detected header as column names
        header_values = df.iloc[header_row].values
        data_df = df.iloc[header_row+1:].copy().reset_index(drop=True)
        
        # Assign header values as column names where possible
        for i, header in enumerate(header_values):
            if i < len(data_df.columns) and not pd.isna(header) and str(header).strip():
                data_df.rename(columns={data_df.columns[i]: str(header)}, inplace=True)
    else:
        st.info("No specific header row detected. Using all data rows.")
        data_df = df.copy()
    
    # Step 2: Find date and value columns
    date_col, value_col = identify_date_value_columns(data_df)
    
    if date_col is not None and value_col is not None:
        st.success(f"✅ Found date column at index {date_col} and value column at index {value_col}")
        
        # Create result dataframe with only the necessary columns
        result_df = pd.DataFrame({
            'date': pd.to_datetime(data_df.iloc[:, date_col], errors='coerce'),
            'value': pd.to_numeric(
                data_df.iloc[:, value_col].astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                errors='coerce'
            )
        })
        
        # Drop rows with NaN values
        result_df = result_df.dropna()
        
        # Sort by date
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        if len(result_df) > 0:
            st.success(f"✅ Successfully extracted {len(result_df)} data points!")
            st.subheader("Extracted Time Series Data")
            st.dataframe(result_df)
            return result_df, 'date', 'value'
    
    # If we reach here, extraction failed
    st.error("❌ Could not extract time series data")
    return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'

def detect_header_row(df: pd.DataFrame) -> int:
    """Detect which row contains the header information."""
    # Check first 10 rows (or less if table is smaller)
    max_rows = min(10, len(df))
    
    # Track potential header rows with scores
    header_scores = {}
    
    for i in range(max_rows):
        row = df.iloc[i]
        score = 0
        
        # Check if row has keywords that suggest it's a header
        header_keywords = ['date', 'price', 'value', 'time', 'period', 'quantity', 
                          'country', 'region', 'store', 'product', 'cost', 'sales']
        
        for val in row:
            val_str = str(val).lower() if not pd.isna(val) else ""
            if any(keyword in val_str for keyword in header_keywords):
                score += 3
                
            # Headers usually don't have many digits
            digit_count = sum(c.isdigit() for c in val_str)
            if digit_count > 0:
                score -= 1
                
            # Headers typically have text
            if len(val_str) > 0 and any(c.isalpha() for c in val_str):
                score += 1
                
        # Store score for this row
        header_scores[i] = score
    
    # Find row with highest score
    if header_scores:
        best_header = max(header_scores.items(), key=lambda x: x[1])
        if best_header[1] > 0:  # Only return if score is positive
            return best_header[0]
    
    return None

def identify_date_value_columns(df: pd.DataFrame) -> Tuple[int, int]:
    """Identify which columns contain date and value (numeric) data."""
    # Check each column for date and numeric content
    date_candidates = []
    numeric_candidates = []
    
    for i, col_name in enumerate(df.columns):
        col = df.iloc[:, i]
        
        # Check if column name suggests it's a date
        col_name_str = str(col_name).lower()
        date_name_score = 0
        if any(date_term in col_name_str for date_term in ['date', 'time', 'period', 'month', 'year', 'updated']):
            date_name_score = 3
        
        # Check if values look like dates
        date_count = 0
        total_values = 0
        for val in col:
            if pd.isna(val):
                continue
                
            total_values += 1
            val_str = str(val)
            
            # Count date-like patterns
            if re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', val_str) or re.search(r'\d{1,2}[-/\.][A-Za-z]{3}[-/\.]\d{2,4}', val_str):
                date_count += 1
                
        date_value_score = date_count / total_values if total_values > 0 else 0
        date_candidates.append((i, date_name_score + date_value_score * 5))
        
        # Check for numeric values
        numeric_count = 0
        for val in col:
            if pd.isna(val):
                continue
                
            # Check if value is numeric or contains dollar/euro/etc.
            val_str = str(val)
            if isinstance(val, (int, float)) or (isinstance(val, str) and 
                (any(c.isdigit() for c in val_str) and any(c in val_str for c in ['$', '€', '¥', '£']))):
                numeric_count += 1
                
        # Check if column name suggests it's a value
        value_name_score = 0
        if any(value_term in col_name_str for value_term in ['price', 'value', 'cost', 'amount', 'quantity', 'sales', 'usd', 'eur']):
            value_name_score = 3
                
        numeric_value_score = numeric_count / total_values if total_values > 0 else 0
        numeric_candidates.append((i, value_name_score + numeric_value_score * 5))
    
    # Sort candidates by score (descending)
    date_candidates.sort(key=lambda x: x[1], reverse=True)
    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Pick best candidates that aren't the same column
    date_col = date_candidates[0][0] if date_candidates else None
    
    # For numeric column, skip any that match the date column
    for num_col, score in numeric_candidates:
        if num_col != date_col:
            value_col = num_col
            break
    else:
        value_col = None
    
    # If no date column found but we have numeric columns, use first column as date
    if date_col is None and numeric_candidates:
        date_col = 0
        
    # If no value column found but we have date column, use the column with most numeric values
    if value_col is None and date_col is not None:
        for i, col_name in enumerate(df.columns):
            if i != date_col:
                # Count numeric values
                numeric_count = sum(1 for val in df.iloc[:, i] 
                                  if isinstance(val, (int, float)) or 
                                  (isinstance(val, str) and any(c.isdigit() for c in str(val))))
                if numeric_count > 0:
                    value_col = i
                    break
    
    return date_col, value_col
