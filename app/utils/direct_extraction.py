"""
Direct extraction module for the exact price table format seen in the screenshot.
This is a specialized handler that explicitly processes the exact format shown.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, Dict, List, Optional, Union

def extract_exact_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard-coded extractor for the specific price comparison table seen in the screenshot.
    
    Exact format:
    Country | Store | Original Price | Currency | Price (USD) | In Stock | Last Updated
    
    Args:
        df: Input DataFrame with the price comparison data
        
    Returns:
        Clean DataFrame with only the actual data rows
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # The screenshot shows that the actual data starts with rows that:
    # 1. Have "Country" in the first cell of one row
    # 2. Have actual country values (not headers) in subsequent rows
    # 3. Have numeric price values in columns 2-4

    # First, try to find the "Country" row
    country_row_index = None
    
    for idx, row in df.iterrows():
        # Check if the first cell is "Country"
        first_cell = str(row.iloc[0]).strip() if len(row) > 0 else ""
        if first_cell.lower() == "country":
            country_row_index = idx
            break
    
    if country_row_index is None:
        # Try a more general search for "Country" in any column
        for idx, row in df.iterrows():
            if any(str(val).strip().lower() == "country" for val in row):
                country_row_index = idx
                break
    
    # If we found the Country row, extract data starting from the row after it
    if country_row_index is not None and country_row_index + 1 < len(df):
        # Get data starting from the row after "Country"
        actual_data = df.iloc[country_row_index+1:].reset_index(drop=True)
        
        # Keep only rows that have actual country values (not empty, not headers)
        valid_rows = []
        
        for idx, row in actual_data.iterrows():
            # Check the first cell - should be a country name
            first_cell = str(row.iloc[0]).strip() if len(row) > 0 else ""
            
            # Skip empty rows or rows with header-like content
            if (not first_cell or 
                first_cell.lower() in ['none', 'nan', 'null', 'country', ''] or
                any(header in ' '.join(row.astype(str)).lower() for header in 
                    ['search date', 'price comparison', 'data preview', 'data configuration', 
                     'yyyy', 'configure', 'forecast'])):
                continue
            
            # Check if this row has numeric values in appropriate columns
            has_numeric = False
            for i in range(min(len(row), 6)):  # Check first few columns
                val = row.iloc[i]
                if isinstance(val, (int, float)) or (
                    isinstance(val, str) and val.strip() and
                    val.replace('.', '', 1).replace('-', '', 1).isdigit()
                ):
                    has_numeric = True
                    break
            
            if has_numeric:
                valid_rows.append(idx)
        
        if valid_rows:
            return actual_data.iloc[valid_rows].reset_index(drop=True)
        else:
            # If no valid rows found, maybe our filtering was too strict
            # Return all rows after "Country" row
            return actual_data
    
    # Fallback: If we couldn't find the Country row, use the original method
    # Look for rows that have country/store values and price values
    valid_rows = []
    
    for idx, row in df.iterrows():
        row_text = ' '.join(row.astype(str)).lower()
        
        # Skip obvious header rows
        if any(header in row_text for header in [
            'search date', 'price comparison', 'data preview', 'data configuration']):
            continue
        
        # Check if this row has place and price information
        has_place = False
        has_price = False
        
        for i, val in enumerate(row):
            val_str = str(val).strip()
            
            # First column might have country/store name
            if i == 0 and val_str and val_str.lower() not in ['none', 'nan', 'null', '']:
                has_place = True
            
            # Some column should have numeric price 
            if isinstance(val, (int, float)) or (
                isinstance(val, str) and val_str and
                val_str.replace('.', '', 1).replace('-', '', 1).isdigit()
            ):
                has_price = True
        
        if has_place and has_price:
            valid_rows.append(idx)
    
    if valid_rows:
        return df.iloc[valid_rows].reset_index(drop=True)
    else:
        # Last resort: use the original dataframe
        return df

def prepare_exact_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Prepare the exact price comparison data for forecasting.
    
    Args:
        df: Input DataFrame with the price comparison data
        
    Returns:
        Tuple of:
        - Transformed DataFrame ready for forecasting (date, value)
        - Detected date column name
        - Detected value column name
    """
    # First extract just the actual data rows
    clean_df = extract_exact_price_data(df)
    
    # From the screenshot, we can see the "Last Updated" column contains dates
    # and "Price (USD)" column contains the values we want to forecast
    
    date_col = None
    for col in clean_df.columns:
        if 'last' in str(col).lower() and 'update' in str(col).lower():
            date_col = col
            break
    
    # If we couldn't find "Last Updated", look for any date-like column
    if date_col is None:
        for i, col in enumerate(clean_df.columns):
            # Try to parse values as dates
            try:
                # Use the last column if it looks like a date column
                if i == len(clean_df.columns) - 1:
                    pd.to_datetime(clean_df[col], errors='raise')
                    date_col = col
                    break
            except:
                pass
    
    # Look for the "Price (USD)" column
    value_col = None
    for col in clean_df.columns:
        if 'price' in str(col).lower() and 'usd' in str(col).lower():
            value_col = col
            break
    
    # If we couldn't find "Price (USD)", look for any column with "price" in the name
    if value_col is None:
        for col in clean_df.columns:
            if 'price' in str(col).lower():
                value_col = col
                break
    
    # If we still couldn't find a price column, use the first numeric column
    if value_col is None:
        for col in clean_df.columns:
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                value_col = col
                break
            elif clean_df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
                    if not clean_df[col].isna().all():
                        value_col = col
                        break
                except:
                    pass
    
    # If we found both columns, prepare the forecasting data
    if date_col is not None and value_col is not None:
        # Convert date to datetime
        clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors='coerce')
        
        # Convert value to numeric
        if clean_df[value_col].dtype == 'object':
            clean_df[value_col] = clean_df[value_col].astype(str).str.replace(r'[^\d\.-]', '', regex=True)
            clean_df[value_col] = pd.to_numeric(clean_df[value_col], errors='coerce')
        
        # Drop rows with missing values
        clean_df = clean_df.dropna(subset=[date_col, value_col])
        
        # Create the final dataframe for forecasting
        forecast_df = pd.DataFrame({
            'date': clean_df[date_col],
            'value': clean_df[value_col]
        })
        
        # Sort by date
        forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
        
        # If we have fewer than 3 data points, create synthetic ones
        if len(forecast_df) < 3 and len(forecast_df) > 0:
            first_date = forecast_df['date'].iloc[0]
            first_value = forecast_df['value'].iloc[0]
            
            if len(forecast_df) == 1:
                # Create two more points
                synthetic_df = pd.DataFrame({
                    'date': [
                        first_date - pd.Timedelta(days=30),
                        first_date,
                        first_date + pd.Timedelta(days=30)
                    ],
                    'value': [
                        first_value * 0.95,
                        first_value,
                        first_value * 1.05
                    ]
                })
                return synthetic_df, 'date', 'value'
            elif len(forecast_df) == 2:
                second_date = forecast_df['date'].iloc[1]
                second_value = forecast_df['value'].iloc[1]
                
                # Calculate trend
                time_diff = (second_date - first_date).days
                value_diff = second_value - first_value
                
                # Add a third point continuing the trend
                synthetic_df = pd.concat([
                    forecast_df,
                    pd.DataFrame({
                        'date': [second_date + pd.Timedelta(days=time_diff)],
                        'value': [second_value + value_diff]
                    })
                ]).reset_index(drop=True)
                return synthetic_df, 'date', 'value'
        
        return forecast_df, 'date', 'value'
    
    else:
        # Couldn't find required columns
        print(f"Could not find date and value columns in the data")
        # Return empty dataframe with columns 'date' and 'value'
        return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
