"""
Simple country row extractor that finds the 'Country' row and extracts data after it.
Specifically designed for price comparison tables.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple

def extract_country_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Extract data by finding the 'Country' row and using rows after it.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of:
        - DataFrame with date and price columns
        - Date column name
        - Value column name
    """
    # Show the original data first
    st.subheader("Original Data")
    st.dataframe(df.head(10))
    
    # Step 1: Find the row that contains "Country" in the first column
    country_row = None
    for idx, row in df.iterrows():
        if idx < len(df) and len(row) > 0:
            first_val = str(row.iloc[0]).strip().lower() if not pd.isna(row.iloc[0]) else ""
            if first_val == "country":
                country_row = idx
                st.success(f"✅ Found Country row at index {idx}")
                break
    
    # If not found by first column, try looking for "Country" in any cell
    if country_row is None:
        for idx, row in df.iterrows():
            for cell in row:
                if isinstance(cell, str) and cell.strip().lower() == "country":
                    country_row = idx
                    st.success(f"✅ Found Country row at index {idx}")
                    break
            if country_row is not None:
                break
    
    if country_row is not None:
        # Get column headers from the Country row
        headers = [str(x) for x in df.iloc[country_row].values]
        
        # Get data rows (everything after the Country row)
        data_rows = df.iloc[country_row+1:].reset_index(drop=True)
        
        # Print information for debugging
        st.write(f"Headers from Country row: {headers}")
        st.write(f"Found {len(data_rows)} data rows after the Country row")
        
        # Set column names if they exist
        if len(headers) == len(data_rows.columns):
            data_rows.columns = headers
        
        # Show the extracted rows for debugging
        st.subheader("Extracted Rows (after Country row)")
        st.dataframe(data_rows)
        
        # Find price and date columns
        price_col_idx = None
        date_col_idx = None
        
        # Look through column names for price and date columns
        for i, col in enumerate(headers):
            col_lower = str(col).lower()
            if "price" in col_lower and "usd" in col_lower:
                price_col_idx = i
            elif "last" in col_lower and "updated" in col_lower:
                date_col_idx = i
        
        # If not found in headers, use defaults
        if price_col_idx is None:
            # Look for column with numeric values (most likely the price)
            for i in range(len(headers)):
                if i != date_col_idx:  # Skip date column if found
                    numeric_count = sum(1 for val in data_rows.iloc[:, i] 
                                     if isinstance(val, (int, float)) or
                                     (isinstance(val, str) and 
                                      any(c.isdigit() for c in val)))
                    if numeric_count > 0:
                        price_col_idx = i
                        break
        
        if date_col_idx is None:
            # Default to last column for date
            date_col_idx = len(headers) - 1
        
        st.write(f"Using column {price_col_idx} for price and column {date_col_idx} for date")
        
        # Filter out title rows, empty rows, and non-data rows
        valid_rows = []
        for idx, row in data_rows.iterrows():
            # Skip rows that look like titles (contain "Price Comparison" or similar)
            first_col_value = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ""
            if "price comparison" in first_col_value or "redmi" in first_col_value:
                st.warning(f"Skipping title row: {row.iloc[0]}")
                continue
            
            # Get values from price and date columns
            price_val = row.iloc[price_col_idx] if price_col_idx < len(row) else None
            date_val = row.iloc[date_col_idx] if date_col_idx < len(row) else None
            
            # Check if row has non-empty values
            if not pd.isna(price_val) and not pd.isna(date_val):
                # Ensure the first column contains a country name (not "Price Comparison" or other header text)
                first_col = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ""
                if len(first_col) > 0 and not any(x in first_col for x in ["comparison", "search", "none", "unnamed"]):
                    # Ensure price is numeric or contains digits
                    if isinstance(price_val, (int, float)) or (isinstance(price_val, str) and any(c.isdigit() for c in price_val)):
                        valid_rows.append(idx)
        
        # Create the result dataframe
        if valid_rows:
            extracted_price = data_rows.iloc[valid_rows, price_col_idx]
            extracted_date = data_rows.iloc[valid_rows, date_col_idx]
            
            # Convert to proper types
            # For prices, extract numeric values
            prices_numeric = pd.to_numeric(
                pd.Series(extracted_price).astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                errors='coerce'
            )
            
            # For dates, convert to datetime
            dates_dt = pd.to_datetime(extracted_date, errors='coerce')
            
            # Create the final dataframe
            result_df = pd.DataFrame({
                'date': dates_dt,
                'value': prices_numeric
            })
            
            # Drop rows with NaN values
            result_df = result_df.dropna()
            
            # Sort by date
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            if len(result_df) > 0:
                st.success(f"✅ Successfully extracted {len(result_df)} rows with price data!")
                st.subheader("Final Extracted Data for Forecasting")
                st.dataframe(result_df)
                return result_df, 'date', 'value'
    
    # If we get here, we couldn't extract data using the Country row method
    # As a fallback, try to extract data by explicitly looking for rows with countries
    st.warning("Could not extract data using Country row. Trying alternative method...")
    
    # Define common country names to look for
    country_names = ["china", "india", "usa", "brazil", "russia", "japan", "germany", 
                     "uk", "france", "italy", "spain", "mexico", "indonesia", "pakistan"]
    
    # Look for rows where the first column contains a country name
    country_rows = []
    for idx, row in df.iterrows():
        first_col = str(row.iloc[0]).lower() if not pd.isna(row.iloc[0]) else ""
        if any(country in first_col for country in country_names):
            country_rows.append(idx)
    
    if country_rows:
        st.success(f"Found {len(country_rows)} rows with country names")
        
        # Get just these rows
        filtered_df = df.iloc[country_rows].reset_index(drop=True)
        st.subheader("Filtered Rows with Country Names")
        st.dataframe(filtered_df)
        
        # Find columns with price and date values
        price_col = None
        date_col = None
        
        # Check every column for number content (likely price data)
        numeric_cols = []
        for col in range(filtered_df.shape[1]):
            # Count how many numeric values in this column
            numeric_count = sum(1 for val in filtered_df.iloc[:, col] 
                            if isinstance(val, (int, float)) or 
                              (isinstance(val, str) and any(c.isdigit() for c in str(val))))
            
            if numeric_count > 0:
                numeric_cols.append((col, numeric_count))
        
        # Sort by count (descending)
        numeric_cols.sort(key=lambda x: x[1], reverse=True)
        
        if numeric_cols:
            # Use column with most numeric values for price
            price_col = numeric_cols[0][0]
            
            # Look for date column (typically has date format or "Updated" in header)
            for col in range(filtered_df.shape[1]):
                if col != price_col:  # Skip price column
                    header = str(df.iloc[0, col]).lower() if not pd.isna(df.iloc[0, col]) else ""
                    if "date" in header or "updated" in header:
                        date_col = col
                        break
            
            # If date column not found, use the last column
            if date_col is None:
                date_col = filtered_df.shape[1] - 1
            
            st.write(f"Using column {price_col} for price and column {date_col} for date")
            
            # Extract values
            extracted_price = filtered_df.iloc[:, price_col]
            extracted_date = filtered_df.iloc[:, date_col]
            
            # Convert to proper types
            prices_numeric = pd.to_numeric(
                pd.Series(extracted_price).astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                errors='coerce'
            )
            
            dates_dt = pd.to_datetime(extracted_date, errors='coerce')
            
            # Create the final dataframe
            result_df = pd.DataFrame({
                'date': dates_dt,
                'value': prices_numeric
            })
            
            # Drop rows with NaN values
            result_df = result_df.dropna()
            
            # Sort by date
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            if len(result_df) > 0:
                st.success(f"✅ Successfully extracted {len(result_df)} rows with country price data!")
                st.subheader("Final Extracted Data for Forecasting")
                st.dataframe(result_df)
                return result_df, 'date', 'value'
    
    # If we reach here, all approaches failed
    st.error("❌ Could not extract data using any method")
    return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'
