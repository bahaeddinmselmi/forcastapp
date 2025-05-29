"""
Direct hardcoded extractor for Redmi Buds price comparison tables.
No fancy algorithms - just works with the exact format from the screenshot.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple

def extract_redmi_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Directly extract price data from Redmi Buds price comparison tables.
    Specifically designed for the exact format in the screenshot.
    
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
    
    st.info("🎯 Using direct extractor specifically designed for Redmi Buds price tables")
    
    # STEP 1: Find the row that contains "Country" in the first column
    country_row = None
    for idx, row in df.iterrows():
        if idx < len(row) and isinstance(row.iloc[0], str) and 'country' in row.iloc[0].lower():
            country_row = idx
            break
    
    if country_row is None:
        st.error("❌ Could not find the 'Country' header row")
        return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
    
    # STEP 2: Extract column names from the Country row
    headers = [str(val) for val in df.iloc[country_row].values]
    
    # Find indices for Date and Price columns
    date_idx = -1  # Default to last column
    price_idx = None
    
    for i, header in enumerate(headers):
        header_lower = header.lower()
        if 'price' in header_lower and 'usd' in header_lower:
            price_idx = i
        elif 'last' in header_lower and 'updated' in header_lower:
            date_idx = i
    
    if price_idx is None:
        st.error("❌ Could not find the Price (USD) column")
        return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
    
    # STEP 3: Extract only the rows that come after the Country row
    data_rows = df.iloc[country_row+1:].reset_index(drop=True)
    
    # STEP 4: Filter out rows that don't have a country name
    valid_rows = []
    for idx, row in data_rows.iterrows():
        if idx < len(row) and isinstance(row.iloc[0], str) and len(row.iloc[0].strip()) > 0:
            # Make sure it's not "None" or "NaN" or empty
            country_value = row.iloc[0].strip().lower()
            if country_value and country_value not in ['none', 'nan', '']:
                valid_rows.append(idx)
    
    # STEP 5: Create the final dataframe
    if valid_rows:
        filtered_data = data_rows.iloc[valid_rows].reset_index(drop=True)
        
        # Extract date and price columns
        dates = filtered_data.iloc[:, date_idx]
        prices = filtered_data.iloc[:, price_idx]
        
        # Convert to proper types
        dates_converted = pd.to_datetime(dates, errors='coerce')
        prices_converted = pd.to_numeric(prices.astype(str).str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')
        
        # Create the result dataframe
        result_df = pd.DataFrame({
            'date': dates_converted,
            'value': prices_converted
        })
        
        # Drop any rows with NaN values
        result_df = result_df.dropna()
        
        if len(result_df) > 0:
            st.success(f"✅ Successfully extracted {len(result_df)} country price points")
            # Show the resulting dataframe
            st.write("Extracted Data:")
            st.dataframe(result_df)
            return result_df, 'date', 'value'
    
    st.error("❌ Could not extract valid rows from the table")
    return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
