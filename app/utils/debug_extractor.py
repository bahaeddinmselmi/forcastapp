"""
Debug extractor that prints diagnostics while extracting data from price tables.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, Dict, List, Optional, Union

def debug_extract_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Debug extractor that prints details of what it's analyzing.
    
    Args:
        df: Input DataFrame with the price comparison data
        
    Returns:
        Clean DataFrame with only the actual data rows
    """
    print("\n==== DEBUG EXTRACTION STARTED ====")
    print(f"Original df shape: {df.shape}")
    
    # Print column names
    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
    
    # Print first 10 rows for debugging
    print("\nFirst 10 rows:")
    for i in range(min(10, len(df))):
        values = [str(val) for val in df.iloc[i].values]
        print(f"  Row {i}: {values}")
    
    # DIRECT APPROACH: Hard-code indices of rows to keep
    # Using extremely specific logic for the exact format in screenshot
    
    # Find rows that start with "Country" (case insensitive)
    country_row_indices = []
    for i in range(len(df)):
        first_val = str(df.iloc[i, 0]).strip().lower() if df.shape[1] > 0 else ""
        if first_val == "country":
            country_row_indices.append(i)
            print(f"\nFound 'Country' row at index {i}")
    
    # If we found a Country row, keep only rows after it that have valid data
    if country_row_indices:
        country_idx = country_row_indices[0]  # Use the first one
        
        # Skip the Country row itself and only keep rows after it
        # Check each row after 'Country' to determine if it's valid data
        valid_rows = []
        
        for i in range(country_idx + 1, len(df)):
            row = df.iloc[i]
            first_val = str(row.iloc[0]).strip().lower() if df.shape[1] > 0 else ""
            
            # Skip rows with empty or header-like first values
            if (not first_val or 
                first_val in ['none', 'nan', 'null', 'search date', ''] or
                first_val.startswith('price comp')):
                print(f"  Skipping row {i}: First value is '{first_val}'")
                continue
            
            # At least one column should have numeric price data
            has_numeric = False
            for j in range(min(len(row), 6)):
                val = row.iloc[j]
                if (isinstance(val, (int, float)) or (
                    isinstance(val, str) and val.strip() and
                    val.replace('.', '', 1).replace('-', '', 1).isdigit()
                )):
                    has_numeric = True
                    break
            
            if has_numeric:
                print(f"  Keeping row {i} with data: {[str(val) for val in row.values]}")
                valid_rows.append(i)
            else:
                print(f"  Skipping row {i}: No numeric data found")
        
        print(f"\nFound {len(valid_rows)} valid data rows")
        
        if valid_rows:
            result_df = df.iloc[valid_rows].reset_index(drop=True)
            print(f"Result df shape: {result_df.shape}")
            return result_df
        else:
            print("No valid data rows found after Country row")
            return pd.DataFrame()  # Empty DataFrame
    
    # If we can't find the Country row, try a simple approach of
    # looking for rows with specific column patterns
    print("\nNo Country row found, trying alternative detection")
    
    valid_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Check for rows that look like they contain country/store and price data
        has_place = False
        has_price = False
        
        for j, val in enumerate(row):
            # First column should have country/store
            if j == 0 and isinstance(val, str) and val.strip() and val.strip().lower() not in ['none', 'nan', 'null', '']:
                has_place = True
            
            # Some column should have numeric data
            if isinstance(val, (int, float)) or (
                isinstance(val, str) and val.strip() and
                val.replace('.', '', 1).replace('-', '', 1).isdigit()
            ):
                has_price = True
                
        # Skip obvious header rows
        row_text = ' '.join([str(val).lower() for val in row])
        is_header = any(term in row_text for term in ['search date', 'price comparison', 'data preview'])
        
        if has_place and has_price and not is_header:
            valid_rows.append(i)
            print(f"  Alternative: Keeping row {i}")
    
    if valid_rows:
        result_df = df.iloc[valid_rows].reset_index(drop=True)
        print(f"Result df shape: {result_df.shape}")
        return result_df
    else:
        print("No valid data rows found with alternative detection")
        return pd.DataFrame()  # Empty DataFrame

def debug_prepare_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Debug version of price data preparation for forecasting.
    
    Args:
        df: Input DataFrame with the price comparison data
        
    Returns:
        Tuple of:
        - Transformed DataFrame ready for forecasting (date, value)
        - Detected date column name
        - Detected value column name
    """
    # First extract just the actual data rows with debugging
    clean_df = debug_extract_price_data(df)
    
    if clean_df.empty:
        print("No data rows found, cannot prepare for forecasting")
        return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
    
    print("\n==== PREPARING DATA FOR FORECASTING ====")
    
    # Print column names of cleaned data
    print("\nCleaned columns:")
    for i, col in enumerate(clean_df.columns):
        print(f"  {i}: {col}")
    
    # Hard-code column indices based on the screenshot pattern
    # The screenshot shows:
    # 0: Country, 1: Store, 2: Original Price, 3: Currency, 4: Price (USD), 5: In Stock, 6: Last Updated
    
    # Date is the last column (Last Updated)
    date_col = clean_df.columns[-1] if len(clean_df.columns) > 0 else None
    print(f"\nUsing date column: {date_col}")
    
    # Try to find Price (USD) column
    value_col = None
    for i, col in enumerate(clean_df.columns):
        if 'price' in str(col).lower() and 'usd' in str(col).lower():
            value_col = col
            print(f"Found USD price column: {col}")
            break
    
    # If not found, use column at index 4 if it exists (Price USD in screenshot)
    if value_col is None and len(clean_df.columns) > 4:
        value_col = clean_df.columns[4]
        print(f"Using column at index 4 as value column: {value_col}")
    
    # If still not found, use any numeric column
    if value_col is None:
        for i, col in enumerate(clean_df.columns):
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                value_col = col
                print(f"Using numeric column as value column: {col}")
                break
            elif clean_df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    nums = pd.to_numeric(clean_df[col], errors='coerce')
                    if not nums.isna().all():
                        value_col = col
                        clean_df[col] = nums
                        print(f"Converted column to numeric: {col}")
                        break
                except:
                    pass
    
    print(f"Using value column: {value_col}")
    
    # If we found both columns, prepare the forecasting data
    if date_col is not None and value_col is not None:
        # Convert date to datetime
        try:
            clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors='coerce')
            print(f"Converted dates: {clean_df[date_col].tolist()}")
        except Exception as e:
            print(f"Error converting dates: {str(e)}")
        
        # Convert value to numeric if needed
        if not pd.api.types.is_numeric_dtype(clean_df[value_col]):
            try:
                # Strip any currency symbols
                if clean_df[value_col].dtype == 'object':
                    clean_df[value_col] = clean_df[value_col].astype(str).str.replace(r'[^\d\.-]', '', regex=True)
                
                clean_df[value_col] = pd.to_numeric(clean_df[value_col], errors='coerce')
                print(f"Converted values: {clean_df[value_col].tolist()}")
            except Exception as e:
                print(f"Error converting values: {str(e)}")
        
        # Drop rows with missing values
        clean_df = clean_df.dropna(subset=[date_col, value_col])
        
        if clean_df.empty:
            print("No valid rows after converting dates and values")
            return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
        
        # Create the final dataframe for forecasting
        forecast_df = pd.DataFrame({
            'date': clean_df[date_col],
            'value': clean_df[value_col]
        })
        
        # Print the final dataset
        print("\nFinal forecast dataset:")
        for i, row in forecast_df.iterrows():
            print(f"  {row['date']} -> {row['value']}")
        
        # Sort by date
        forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
        
        print(f"\nFinal dataset has {len(forecast_df)} rows")
        
        return forecast_df, 'date', 'value'
    
    else:
        # Couldn't find required columns
        missing = []
        if date_col is None:
            missing.append("date column")
        if value_col is None:
            missing.append("value column")
            
        print(f"Could not identify required {' and '.join(missing)}")
        
        # Return empty dataframe
        return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
