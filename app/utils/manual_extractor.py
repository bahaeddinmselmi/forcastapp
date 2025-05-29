"""
Ultra-simple extractor that directly uses manual input to extract price data.
"""

import pandas as pd
import numpy as np
import streamlit as st
import re
from typing import Tuple

def extract_manual_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Directly extract price data using manual input from the user.
    
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
    
    # Check if we have user-configured settings in the session state
    if 'skip_rows' in st.session_state and 'first_row_is_header' in st.session_state:
        # Use the settings the user has already configured in the UI
        skip_rows = st.session_state.get('skip_rows', 0)
        first_row_is_header = st.session_state.get('first_row_is_header', False)
        st.success(f"✅ Using your configured settings: Skip {skip_rows} rows, {'Using' if first_row_is_header else 'Not using'} first row as header")
        
        # Calculate the effective header row based on user settings
        header_row = skip_rows if first_row_is_header else None
        data_start_row = skip_rows + 1 if first_row_is_header else skip_rows
        
        st.write("📊 Your data table with applied settings:")
        # Show the relevant portion of the table based on user settings
        if header_row is not None:
            # If header row is specified, show it and some data rows
            display_df = pd.DataFrame([df.iloc[header_row].values], columns=df.columns)
            display_df = pd.concat([display_df, df.iloc[data_start_row:data_start_row+10]], ignore_index=True)
            st.dataframe(display_df)
        else:
            # If no header row, just show data rows
            st.dataframe(df.iloc[data_start_row:data_start_row+15])
    else:    
        # Standard detection if user hasn't configured settings
        st.write("📊 Your data table:")
        st.dataframe(df.head(15))
        
        # Detect potential data rows to help the user
        country_row = None
        for idx, row in df.iterrows():
            if idx < len(df) and len(row) > 0:
                first_val = str(row.iloc[0]).strip().lower() if not pd.isna(row.iloc[0]) else ""
                if first_val == "country":
                    country_row = idx
                    st.success(f"✅ Found 'Country' row at index {idx}. This is likely your header row.")
                    break
    
    # Help user find price and date columns
    price_col_guess = None
    date_col_guess = None
    
    # If we have user-configured settings, use those rows for detection
    if 'skip_rows' in st.session_state and 'first_row_is_header' in st.session_state:
        header_row = st.session_state.get('skip_rows', 0) if st.session_state.get('first_row_is_header', False) else None
        data_start_row = st.session_state.get('skip_rows', 0) + 1 if st.session_state.get('first_row_is_header', False) else st.session_state.get('skip_rows', 0)
    else:
        # Use detected values if no user configuration
        header_row = country_row
        data_start_row = country_row + 1 if country_row is not None else 3
    
    # Check for price and date columns using header if available
    if header_row is not None and header_row < len(df):
        for i, val in enumerate(df.iloc[header_row]):
            val_str = str(val).lower() if not pd.isna(val) else ""
            if ('price' in val_str and 'usd' in val_str) or ('price' in val_str):
                price_col_guess = i
                st.info(f"💲 Found price column at index {i}: '{val_str}'")
            elif ('last' in val_str and 'updated' in val_str) or 'date' in val_str:
                date_col_guess = i
                st.info(f"📅 Found date column at index {i}: '{val_str}'")
    
    # Check first data row for numeric values as fallback
    if (price_col_guess is None or date_col_guess is None) and data_start_row < len(df):
        # Check for column with dates
        if date_col_guess is None:
            for i, val in enumerate(df.iloc[data_start_row]):
                val_str = str(val).lower() if not pd.isna(val) else ""
                if re.search(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', val_str) or 'may' in val_str or 'jun' in val_str:
                    date_col_guess = i
                    st.info(f"📅 Found date column at index {i} (based on date format)")
                    break
        
        # Check for column with numbers (likely price)
        if price_col_guess is None:
            for i, val in enumerate(df.iloc[data_start_row]):
                if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r'\d+\.?\d*', val_str)):
                    # Skip if this is the date column
                    if i != date_col_guess:
                        price_col_guess = i
                        st.info(f"💲 Found price column at index {i} (based on numeric format)")
                        break
    
    # Use user-configured start row or detected value
    if 'skip_rows' in st.session_state:
        recommended_start = st.session_state.get('skip_rows', 0)
        if st.session_state.get('first_row_is_header', False):
            recommended_start += 1
    else:
        recommended_start = data_start_row
    
    # Give the user direct control over extraction
    st.markdown("### Manual Data Extraction")
    st.info("📝 Please specify which rows contain your price data and which columns to use.")
    
    # Allow user to input row range
    col1, col2 = st.columns(2)
    with col1:
        start_row = st.number_input("Start Row (0-based index)", 
                            min_value=0, 
                            max_value=len(df)-1, 
                            value=min(recommended_start, len(df)-1),
                            help="This is the first row that contains actual data (after headers)")
    with col2:
        end_row = st.number_input("End Row (0-based index)", 
                           min_value=start_row, 
                           max_value=len(df)-1, 
                           value=min(start_row+5, len(df)-1),
                           help="This is the last row that contains data to extract")
    
    # Allow user to select date and price columns
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Date Column", options=list(range(len(df.columns))), 
                                format_func=lambda x: f"{x}: {df.iloc[max(0, min(start_row, len(df)-1)), x]}",
                                index=date_col_guess if date_col_guess is not None else -1,
                                help="Column that contains dates (Last Updated)")
    with col2:
        price_col = st.selectbox("Price Column", options=list(range(len(df.columns))),
                                format_func=lambda x: f"{x}: {df.iloc[max(0, min(start_row, len(df)-1)), x]}",
                                index=price_col_guess if price_col_guess is not None else -1,
                                help="Column that contains prices (Price USD)")
    
    # Extract the selected rows and columns
    if st.button("Extract Selected Data"):
        if start_row <= end_row and 0 <= date_col < len(df.columns) and 0 <= price_col < len(df.columns):
            data_rows = df.iloc[start_row:end_row+1].reset_index(drop=True)
            
            # Create the result dataframe
            dates = data_rows.iloc[:, date_col]
            prices = data_rows.iloc[:, price_col]
            
            # Convert to proper types
            dates_converted = pd.to_datetime(dates, errors='coerce')
            
            # Convert prices to numeric
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
                st.success(f"✅ Successfully extracted {len(result_df)} price points from rows {start_row} to {end_row}")
                # Show the extracted data
                st.write("Extracted Data for Forecasting:")
                st.dataframe(result_df)
                
                # Store the result in session state so it's available after button press
                st.session_state.manual_extraction_result = (result_df, 'date', 'value')
                return result_df, 'date', 'value'
    
    # If we already have results in session state, return those
    if 'manual_extraction_result' in st.session_state:
        result, date_col, value_col = st.session_state.manual_extraction_result
        if len(result) > 0:
            st.info("ℹ️ Using previously extracted data")
            return result, date_col, value_col
    
    # If we couldn't extract valid data or no button press yet, try a simple fallback
    try:
        # As a fallback, try to find the "Price (USD)" column and "Last Updated" column automatically
        price_col_idx = None
        date_col_idx = None
        
        # First check column names if they exist
        for i, col in enumerate(df.columns):
            col_name = str(col).lower()
            if 'price' in col_name and 'usd' in col_name:
                price_col_idx = i
            elif 'last' in col_name and 'updated' in col_name:
                date_col_idx = i
        
        # If not found in column names, look in the first few rows
        if price_col_idx is None or date_col_idx is None:
            for row_idx in range(min(10, len(df))):
                for col_idx in range(len(df.columns)):
                    cell_value = str(df.iloc[row_idx, col_idx]).lower()
                    if 'price' in cell_value and 'usd' in cell_value and price_col_idx is None:
                        price_col_idx = col_idx
                    elif 'last' in cell_value and 'updated' in cell_value and date_col_idx is None:
                        date_col_idx = col_idx
        
        # If still not found, use last column for dates and try to find prices by numeric values
        if date_col_idx is None:
            date_col_idx = len(df.columns) - 1
        
        if price_col_idx is None:
            for col_idx in range(len(df.columns)):
                if col_idx != date_col_idx:
                    numeric_count = sum(1 for val in df.iloc[:, col_idx] 
                                     if isinstance(val, (int, float)) and not pd.isna(val))
                    if numeric_count >= 3:
                        price_col_idx = col_idx
                        break
        
        # If we found both columns, extract rows with non-empty values in both
        if price_col_idx is not None and date_col_idx is not None:
            valid_rows = []
            for row_idx in range(len(df)):
                date_val = df.iloc[row_idx, date_col_idx]
                price_val = df.iloc[row_idx, price_col_idx]
                
                if not pd.isna(date_val) and not pd.isna(price_val):
                    if isinstance(price_val, (int, float)) or (
                        isinstance(price_val, str) and any(c.isdigit() for c in price_val)
                    ):
                        valid_rows.append(row_idx)
            
            if valid_rows:
                dates = df.iloc[valid_rows, date_col_idx]
                prices = df.iloc[valid_rows, price_col_idx]
                
                result_df = pd.DataFrame({
                    'date': pd.to_datetime(dates, errors='coerce'),
                    'value': pd.to_numeric(prices.astype(str).str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')
                }).dropna()
                
                if len(result_df) > 0:
                    st.info(f"ℹ️ Automatically extracted {len(result_df)} rows as a starting point")
                    st.write("Please use the controls above to adjust if needed.")
                    return result_df, 'date', 'value'
    except Exception as e:
        st.warning(f"Could not auto-extract data: {str(e)}")
    
    # If everything fails, return empty dataframe
    return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
