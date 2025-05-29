"""
Simple numeric-based extractor for price data.
Finds columns with numbers to identify price data.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple

def extract_numeric_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Extract price data by finding columns with numeric values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of:
        - DataFrame with date and price columns
        - Date column name
        - Price column name
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    st.info("🔢 Using simple numeric detection to extract price data")
    
    # STEP 1: Find which columns have numeric values
    numeric_columns = []
    for col_idx in range(len(df.columns)):
        # Check if this column has numeric values
        numeric_count = 0
        for val in df.iloc[:, col_idx]:
            # Try to convert the value to a number
            try:
                if isinstance(val, (int, float)) and not pd.isna(val):
                    numeric_count += 1
                elif isinstance(val, str) and val.strip():
                    # Try to extract a number from the string
                    numeric_val = ''.join(c for c in val if c.isdigit() or c in ['.', '-'])
                    if numeric_val and float(numeric_val) > 0:
                        numeric_count += 1
            except:
                pass
        
        # If at least 3 values are numeric, consider it a numeric column
        if numeric_count >= 3:
            numeric_columns.append(col_idx)
    
    # STEP 2: Find date column (typically the last column)
    date_column = None
    for col_idx in range(len(df.columns)):
        # Check if this column has date-like values
        date_count = 0
        for val in df.iloc[:, col_idx]:
            if isinstance(val, str) and any(month in val.lower() for month in 
                                         ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                date_count += 1
        
        # If at least 3 values look like dates, consider it a date column
        if date_count >= 3:
            date_column = col_idx
            break
    
    # If we couldn't find a date column, use the last column
    if date_column is None:
        date_column = len(df.columns) - 1
    
    # STEP 3: Find the best price column (should be numeric and not the date column)
    price_column = None
    for col_idx in numeric_columns:
        if col_idx != date_column:
            price_column = col_idx
            # Check if it has "price" or "usd" in the column name or header
            for row_idx in range(min(10, len(df))):
                val = str(df.iloc[row_idx, col_idx]).lower()
                if 'price' in val or 'usd' in val:
                    price_column = col_idx
                    break
            # If we found a column with price in the name, stop looking
            if price_column == col_idx:
                break
    
    # If we couldn't find a price column, use the first numeric column that's not the date
    if price_column is None and numeric_columns:
        for col_idx in numeric_columns:
            if col_idx != date_column:
                price_column = col_idx
                break
    
    # STEP 4: If we found both date and price columns, extract the data
    if date_column is not None and price_column is not None:
        # Find rows that have non-empty values in both columns
        valid_rows = []
        for row_idx in range(len(df)):
            date_val = df.iloc[row_idx, date_column]
            price_val = df.iloc[row_idx, price_column]
            
            # Skip empty values
            if pd.isna(date_val) or pd.isna(price_val):
                continue
                
            # Skip non-numeric price values
            try:
                if isinstance(price_val, str):
                    # Try to extract a number from the string
                    numeric_val = ''.join(c for c in price_val if c.isdigit() or c in ['.', '-'])
                    if not numeric_val or float(numeric_val) <= 0:
                        continue
                elif not isinstance(price_val, (int, float)) or price_val <= 0:
                    continue
            except:
                continue
                
            valid_rows.append(row_idx)
        
        # If we found valid rows, create the result dataframe
        if valid_rows:
            dates = df.iloc[valid_rows, date_column]
            prices = df.iloc[valid_rows, price_column]
            
            # Convert to proper types
            dates_converted = pd.to_datetime(dates, errors='coerce')
            
            # Convert prices to numeric, handling cases where they're already numeric
            if prices.dtype in [np.int64, np.float64]:
                prices_converted = prices
            else:
                prices_converted = pd.to_numeric(prices.astype(str).str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')
            
            # Create the result dataframe
            result_df = pd.DataFrame({
                'date': dates_converted,
                'value': prices_converted
            })
            
            # Drop any rows with NaN values
            result_df = result_df.dropna()
            
            if len(result_df) > 0:
                st.success(f"✅ Successfully extracted {len(result_df)} price points by finding numeric columns")
                # Show the extracted data
                st.write("Extracted Data:")
                st.dataframe(result_df)
                return result_df, 'date', 'value'
    
    # If we couldn't extract valid data, return an empty dataframe
    st.error("❌ Could not find valid numeric price data in the table")
    return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
