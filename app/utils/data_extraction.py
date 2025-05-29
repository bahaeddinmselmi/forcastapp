"""
AI-powered data extraction module for transforming various data formats into time series data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, Dict, List, Optional, Union

def extract_time_series_from_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Intelligently extract time series data from various input formats.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        Tuple of:
        - Transformed DataFrame ready for forecasting
        - Detected date column name
        - Detected value column name
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Step 1: Identify potential date columns
    date_cols = []
    for col in df.columns:
        col_name = str(col).lower()
        # Check column name for date-related terms
        if any(term in col_name for term in ['date', 'time', 'day', 'month', 'year', 'updated', 'created']):
            date_cols.append(col)
        
        # Also check content if not already identified
        if col not in date_cols and df[col].dtype == 'object':
            # Sample first 5 non-null values
            sample = df[col].dropna().head(5).astype(str)
            
            # Check if values look like dates
            if all(re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|\d{1,2}[-/\s][A-Za-z]{3}[-/\s]\d{2,4}', str(val)) for val in sample):
                date_cols.append(col)
    
    # Step 2: Identify potential value columns (numeric columns that aren't IDs)
    value_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_name = str(col).lower()
            # Skip likely ID columns
            if any(term in col_name for term in ['id', 'code', 'number', 'zip', 'postal']):
                continue
            
            # Prioritize columns with value-related terms
            priority = 0
            if any(term in col_name for term in ['price', 'value', 'amount', 'quantity', 'demand', 'sales', 'revenue']):
                priority = 3
            elif any(term in col_name for term in ['cost', 'profit', 'margin', 'discount']):
                priority = 2
            elif df[col].nunique() > 5:  # Columns with variety of values
                priority = 1
                
            value_cols.append((col, priority))
    
    # Sort value columns by priority
    value_cols.sort(key=lambda x: x[1], reverse=True)
    value_cols = [col for col, _ in value_cols]
    
    # Step 3: If no clear date column, try to use 'Last Updated' or create a synthetic one
    if not date_cols and 'Last Updated' in df.columns:
        date_cols = ['Last Updated']
    elif not date_cols:
        print("No date column detected. Creating a synthetic date index.")
        df['Generated Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        date_cols = ['Generated Date']
    
    # Step 4: Select the best date and value columns
    selected_date_col = date_cols[0] if date_cols else None
    selected_value_col = value_cols[0] if value_cols else None
    
    # Step 5: Check if there are identifying columns (like Product, Country, Store)
    # We may need to filter or group by these for proper forecasting
    id_cols = []
    for col in df.columns:
        col_name = str(col).lower()
        if any(term in col_name for term in ['product', 'item', 'sku', 'model', 'country', 'region', 'store', 'location']):
            id_cols.append(col)
    
    # Step 6: Transform data based on what we found
    if selected_date_col and selected_value_col:
        # If we have identifying columns, we need to decide how to handle them
        if id_cols:
            print(f"Multiple entities detected in columns: {id_cols}")
            print("Using the first entity for forecasting.")
            # Use the first row's values for identifying columns to filter
            filter_dict = {col: df[col].iloc[0] for col in id_cols}
            
            # Filter to just this entity
            for col, val in filter_dict.items():
                df = df[df[col] == val]
        
        # Final preparation of date column
        try:
            df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
            df = df.dropna(subset=[selected_date_col])
            
            # Extract meaningful data for forecasting
            ts_data = df[[selected_date_col, selected_value_col]].copy()
            ts_data = ts_data.rename(columns={selected_date_col: 'date', selected_value_col: 'value'})
            
            # If too few points for meaningful forecasting, create synthetic ones
            if len(ts_data) < 3:
                print(f"Only {len(ts_data)} data points found. Creating synthetic points.")
                
                # Get existing points
                dates = ts_data['date'].tolist()
                values = ts_data['value'].tolist()
                
                # If only one point, create two more with variations
                if len(ts_data) == 1:
                    base_date = dates[0]
                    base_value = values[0]
                    
                    # Add one point before, one point after
                    ts_data = pd.DataFrame({
                        'date': [
                            base_date - pd.Timedelta(days=30),
                            base_date,
                            base_date + pd.Timedelta(days=30)
                        ],
                        'value': [
                            base_value * 0.95,
                            base_value,
                            base_value * 1.05
                        ]
                    })
                
                # If two points, add a third following the trend
                elif len(ts_data) == 2:
                    date_diff = (dates[1] - dates[0]).days
                    value_diff = values[1] - values[0]
                    
                    ts_data = pd.DataFrame({
                        'date': [
                            dates[0],
                            dates[1],
                            dates[1] + pd.Timedelta(days=date_diff)
                        ],
                        'value': [
                            values[0],
                            values[1],
                            values[1] + value_diff
                        ]
                    })
            
            return ts_data, 'date', 'value'
            
        except Exception as e:
            print(f"Error processing dates: {str(e)}")
            raise ValueError(f"Could not process date column '{selected_date_col}': {str(e)}")
    
    else:
        missing = []
        if not selected_date_col:
            missing.append("date column")
        if not selected_value_col:
            missing.append("value column")
        
        raise ValueError(f"Could not identify required {' and '.join(missing)} for forecasting.")

def extract_data_from_mixed_format(raw_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extract data from various input formats: dataframes, CSV strings, tab-delimited text, etc.
    Intelligently detects and skips header rows to find where the real data starts.
    
    Args:
        raw_data: Input data in various possible formats
        
    Returns:
        Pandas DataFrame with the extracted data
    """
    if isinstance(raw_data, pd.DataFrame):
        # If we already have a DataFrame, check for header rows
        df = raw_data.copy()
        # Try to detect and skip header rows
        return detect_and_skip_headers(df)
    
    if isinstance(raw_data, str):
        # Try different parsing approaches
        try:
            # First try as CSV
            df = pd.read_csv(pd.StringIO(raw_data))
            if len(df.columns) > 1:
                return detect_and_skip_headers(df)
        except:
            pass
        
        try:
            # Try as tab-delimited
            df = pd.read_csv(pd.StringIO(raw_data), sep='\t')
            if len(df.columns) > 1:
                return detect_and_skip_headers(df)
        except:
            pass
        
        try:
            # Try as space-delimited
            df = pd.read_csv(pd.StringIO(raw_data), delim_whitespace=True)
            if len(df.columns) > 1:
                return detect_and_skip_headers(df)
        except:
            pass
            
        # If all else fails, try to intelligently parse the data
        lines = raw_data.strip().split('\n')
        
        # Try to identify where the real data starts by looking for repeated header-like patterns
        data_start_idx = 0
        headers = None
        
        # First try to find where the column headers are
        for i in range(min(10, len(lines))):
            line = lines[i]
            # Check if this line looks like headers (contains multiple text fields)
            if re.search(r'[A-Za-z]{2,}\s+[A-Za-z]{2,}', line):
                # This might be the header line
                potential_headers = re.split(r'\s{2,}|\t|,|\|', line)
                potential_headers = [h.strip() for h in potential_headers if h.strip()]
                
                # Headers should have a reasonable number of columns (at least 2)
                if len(potential_headers) >= 2:
                    headers = potential_headers
                    data_start_idx = i + 1  # Data starts after headers
                    break
        
        # If no headers found, use the first line
        if headers is None and lines:
            headers = re.split(r'\s{2,}|\t|,|\|', lines[0])
            headers = [h.strip() for h in headers if h.strip()]
            data_start_idx = 1
        
        # Now extract the data rows
        rows = []
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
                
            # Determine the delimiter to use based on this line
            for delim in ['\t', '\s{2,}', ',', '\|', ';']:
                if delim == '\s{2,}':
                    values = re.split(r'\s{2,}', line)
                else:
                    if delim in line:
                        values = line.split(delim)
                    else:
                        continue
                
                values = [v.strip() for v in values if v.strip()]
                
                # If this parsing gives a reasonable number of values, use it
                if len(values) >= min(2, len(headers)):
                    # Ensure matching size with headers
                    while len(values) < len(headers):
                        values.append(None)
                    
                    # Truncate if too many values
                    values = values[:len(headers)]
                    
                    rows.append(values)
                    break
        
        # Create DataFrame
        if rows and headers:
            df = pd.DataFrame(rows, columns=headers)
            # Auto-detect numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except:
                    pass  # Keep as string if conversion fails
            return df
        else:
            # Last resort: try standard parsers with different parameters
            try:
                return pd.read_csv(pd.StringIO(raw_data), skiprows=data_start_idx)
            except:
                # Create an empty DataFrame with the detected headers
                if headers:
                    return pd.DataFrame(columns=headers)
                else:
                    return pd.DataFrame()

def detect_and_skip_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and skips header rows in a DataFrame by looking for rows that likely
    contain column names or metadata instead of actual data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with header rows removed
    """
    if len(df) <= 1:  # Not enough rows to analyze
        return df
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check if the first row contains column names repeated in the header
    first_row_matches_cols = sum(df.columns.str.lower() == df.iloc[0].astype(str).str.lower()) > 0
    
    # Check if the first row has a different data type pattern than the rest
    dtype_patterns = []
    for i in range(min(5, len(df))):
        # Create a pattern string of value types: 's' for string, 'n' for numeric, 'd' for date
        pattern = ""
        for col in df.columns:
            val = df.iloc[i][col]
            if pd.isna(val):
                pattern += "x"  # Missing value
            elif isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '', 1).isdigit()):
                pattern += "n"  # Numeric
            elif isinstance(val, str) and re.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', val):
                pattern += "d"  # Date-like
            else:
                pattern += "s"  # String
        dtype_patterns.append(pattern)
    
    # If we have enough rows, check for pattern changes indicating headers
    if len(dtype_patterns) >= 2:
        first_pattern = dtype_patterns[0]
        rest_patterns = dtype_patterns[1:]
        
        # If first row has a different pattern than the rest, it might be a header
        if first_pattern != rest_patterns[0] and all(p == rest_patterns[0] for p in rest_patterns):
            df_copy = df_copy.iloc[1:].reset_index(drop=True)
            first_row_matches_cols = False  # Already handled
    
    # If the first row repeats column names, drop it
    if first_row_matches_cols:
        df_copy = df_copy.iloc[1:].reset_index(drop=True)
    
    # Check for rows with all or mostly text where other rows are numeric
    if len(df_copy) > 1:
        numeric_cols = df_copy.select_dtypes(include=['number']).columns
        rows_to_drop = []
        
        if len(numeric_cols) > 0:
            for i in range(min(3, len(df_copy))):
                # Count how many of the numeric columns have non-numeric values in this row
                non_numeric_count = 0
                for col in numeric_cols:
                    val = df_copy.iloc[i][col]
                    if isinstance(val, str) and not val.replace('.', '', 1).replace('-', '', 1).isdigit():
                        non_numeric_count += 1
                
                # If most numeric columns have text in this row, it's likely a header
                if non_numeric_count > len(numeric_cols) / 2:
                    rows_to_drop.append(i)
        
        if rows_to_drop:
            df_copy = df_copy.drop(rows_to_drop).reset_index(drop=True)
    
    # Attempt to convert columns to appropriate types
    for col in df_copy.columns:
        # Try to convert to numeric
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='raise')
            continue
        except:
            pass
        
        # Try to convert to datetime
        if df_copy[col].dtype == 'object':
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='raise')
            except:
                pass  # Keep as is if conversion fails
    
    return df_copy
    
    # Unknown format
    raise ValueError("Could not parse input data. Please provide data as a DataFrame or string in CSV format.")

def process_data_for_forecasting(raw_data: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str, str]:
    """
    Process raw data and extract time series components for forecasting.
    
    Args:
        raw_data: Input data in various possible formats
        
    Returns:
        Tuple of:
        - Transformed DataFrame ready for forecasting
        - Detected date column name
        - Detected value column name
    """
    try:
        # First, extract data from whatever format we received
        extracted_df = extract_data_from_mixed_format(raw_data)
        
        # Then, transform it into time series format
        ts_data, date_col, value_col = extract_time_series_from_data(extracted_df)
        
        return ts_data, date_col, value_col
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise ValueError(f"Data processing failed: {str(e)}")
