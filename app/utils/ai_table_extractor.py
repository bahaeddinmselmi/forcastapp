"""
Universal AI-powered table extractor.
Can analyze and extract data from any table structure regardless of format.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Dict, Any, Optional

class AITableExtractor:
    """
    AI-powered table extractor that can intelligently identify and extract
    time series data from any table format.
    """
    
    def __init__(self):
        """Initialize the AI table extractor."""
        self.name = "AI Table Extractor"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
        """
        Extract time series data from any table format using AI-powered analysis.
        
        Args:
            df: Input DataFrame containing the table data
            
        Returns:
            Tuple of:
            - Transformed DataFrame ready for forecasting (date, value columns)
            - Date column name
            - Value column name
        """
        st.info("🧠 AI Table Analyzer is examining your data structure...")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # AI ANALYSIS PHASE 1: Determine table structure and find header row
        # -------------------------------------------------------------
        
        # Find potential header rows (rows that might contain column names)
        potential_headers = []
        
        # Key indicators for header rows
        header_indicators = [
            'country', 'region', 'state', 'province', 'city', 'location',  # Location columns
            'price', 'cost', 'value', 'amount', 'usd', 'eur', 'gbp',      # Price columns
            'date', 'time', 'day', 'month', 'year', 'updated', 'created', # Date columns
            'product', 'item', 'sku', 'model', 'type', 'category',        # Product columns
            'store', 'shop', 'vendor', 'supplier', 'retailer', 'seller'   # Store columns
        ]
        
        # Score each row based on how likely it is to be a header
        header_scores = []
        
        for idx, row in df_copy.iterrows():
            row_text = ' '.join([str(val).lower() for val in row.values if isinstance(val, str)])
            score = sum(2 for indicator in header_indicators if indicator in row_text)
            
            # Extra points for having multiple header indicators in different cells
            indicators_in_different_cells = 0
            for val in row.values:
                if isinstance(val, str):
                    val_lower = val.lower()
                    if any(indicator in val_lower for indicator in header_indicators):
                        indicators_in_different_cells += 1
            
            score += indicators_in_different_cells
            
            # Penalty for rows that are mostly empty
            empty_cells = sum(1 for val in row.values if pd.isna(val) or str(val).strip() == '')
            if empty_cells > len(row) / 2:
                score -= 3
            
            # Rows with many numeric values are less likely to be headers
            numeric_cells = sum(1 for val in row.values if isinstance(val, (int, float)) 
                               or (isinstance(val, str) and val.strip().replace('.', '', 1).isdigit()))
            if numeric_cells > len(row) / 3:
                score -= 2
            
            header_scores.append((idx, score))
        
        # Sort by score in descending order
        header_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Find the best header row (highest score)
        header_row = None
        if header_scores and header_scores[0][1] > 0:
            header_row = header_scores[0][0]
            st.success(f"✅ AI identified likely header row at index {header_row}")
        
        # AI ANALYSIS PHASE 2: Identify date and value columns
        # -------------------------------------------------------------
        
        # If we found a header row, use it to identify columns
        if header_row is not None:
            headers = [str(val).strip() for val in df_copy.iloc[header_row].values]
            
            # Get data rows (everything after the header row)
            data_rows = df_copy.iloc[header_row+1:].reset_index(drop=True)
            
            # Set header as column names if dimensions match
            if len(headers) == data_rows.shape[1]:
                data_rows.columns = headers
            
            # Find date column
            date_col = None
            date_col_scores = {}
            
            for col_idx, col_name in enumerate(data_rows.columns):
                score = 0
                col_name_lower = str(col_name).lower()
                
                # Check column name for date indicators
                if 'date' in col_name_lower:
                    score += 5
                if 'update' in col_name_lower or 'updated' in col_name_lower:
                    score += 8
                if 'time' in col_name_lower or 'day' in col_name_lower:
                    score += 3
                if any(month in col_name_lower for month in 
                      ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                    score += 3
                
                # Check content
                date_format_count = 0
                for val in data_rows.iloc[:, col_idx]:
                    if isinstance(val, str):
                        val_lower = val.lower()
                        # Check for date patterns like "May 12, 2022" or "12/05/2022"
                        if any(month in val_lower for month in 
                              ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                            date_format_count += 1
                        elif '/' in val or '-' in val or '.' in val:
                            parts = val.replace('/', ' ').replace('-', ' ').replace('.', ' ').split()
                            if all(part.isdigit() for part in parts) and 2 <= len(parts) <= 3:
                                date_format_count += 1
                
                score += date_format_count * 2
                
                # Final check: can it be parsed as datetime?
                parseable = pd.to_datetime(data_rows.iloc[:, col_idx], errors='coerce').notna().sum()
                score += parseable
                
                date_col_scores[col_idx] = score
            
            # Find the best date column
            if date_col_scores:
                date_col = max(date_col_scores.items(), key=lambda x: x[1])[0]
            
            # Find value (price) column
            value_col = None
            value_col_scores = {}
            
            for col_idx, col_name in enumerate(data_rows.columns):
                if col_idx == date_col:
                    continue  # Skip date column
                
                score = 0
                col_name_lower = str(col_name).lower()
                
                # Check column name for price indicators
                if 'price' in col_name_lower:
                    score += 5
                    if 'usd' in col_name_lower or '$' in col_name_lower:
                        score += 5
                if 'cost' in col_name_lower or 'value' in col_name_lower:
                    score += 3
                if 'amount' in col_name_lower or 'total' in col_name_lower:
                    score += 2
                
                # Check for currency symbols or codes
                for currency in ['$', '€', '£', 'usd', 'eur', 'gbp', 'jpy', 'cny', 'inr', 'krw']:
                    if currency in col_name_lower:
                        score += 3
                
                # Check content
                numeric_count = 0
                currency_symbol_count = 0
                
                for val in data_rows.iloc[:, col_idx]:
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        numeric_count += 1
                    elif isinstance(val, str):
                        # Check for currency symbols
                        if any(symbol in val for symbol in ['$', '€', '£', '¥']):
                            currency_symbol_count += 1
                        
                        # Check if string contains digits
                        if any(c.isdigit() for c in val):
                            # Try to extract numeric value
                            numeric_str = ''.join(c for c in val if c.isdigit() or c in ['.', '-'])
                            try:
                                float(numeric_str)
                                numeric_count += 1
                            except:
                                pass
                
                score += numeric_count + currency_symbol_count * 2
                
                # Final check: can values be converted to numeric?
                convertible = pd.to_numeric(
                    data_rows.iloc[:, col_idx].astype(str).str.replace(r'[^\d\.-]', '', regex=True), 
                    errors='coerce'
                ).notna().sum()
                
                score += convertible
                
                value_col_scores[col_idx] = score
            
            # Find the best value column
            if value_col_scores:
                value_col = max(value_col_scores.items(), key=lambda x: x[1])[0]
            
            # AI ANALYSIS PHASE 3: Filter out non-data rows
            # -------------------------------------------------------------
            
            # Find rows that actually contain data
            valid_rows = []
            
            # Check if we have a column that looks like a "first column" (Country, Region, etc.)
            first_col_idx = None
            for col_idx, col_name in enumerate(data_rows.columns):
                col_name_lower = str(col_name).lower()
                if any(indicator in col_name_lower for indicator in 
                      ['country', 'region', 'state', 'province', 'city', 'location']):
                    first_col_idx = col_idx
                    break
            
            # Filter rows based on having non-empty values in key columns
            for idx, row in data_rows.iterrows():
                has_date = date_col is not None and not pd.isna(row.iloc[date_col])
                has_value = value_col is not None and not pd.isna(row.iloc[value_col])
                has_first_col = first_col_idx is not None and not pd.isna(row.iloc[first_col_idx])
                
                # Row needs to have either (date and value) or (first_col and value)
                if (has_date and has_value) or (has_first_col and has_value):
                    # Skip rows with "Search Date" or other header indicators
                    row_text = ' '.join([str(val).lower() for val in row.values if isinstance(val, str)])
                    if 'search date' not in row_text and not any(val.lower() == 'date' for val in row.values if isinstance(val, str)):
                        valid_rows.append(idx)
            
            # Create the final dataframe
            if valid_rows and date_col is not None and value_col is not None:
                # Extract the valid rows
                final_data = data_rows.iloc[valid_rows].copy()
                
                # Convert date column to datetime
                final_data.iloc[:, date_col] = pd.to_datetime(final_data.iloc[:, date_col], errors='coerce')
                
                # Convert value column to numeric
                if final_data.iloc[:, value_col].dtype not in [np.int64, np.float64]:
                    final_data.iloc[:, value_col] = pd.to_numeric(
                        final_data.iloc[:, value_col].astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                        errors='coerce'
                    )
                
                # Create the result dataframe
                result_df = pd.DataFrame({
                    'date': final_data.iloc[:, date_col],
                    'value': final_data.iloc[:, value_col]
                })
                
                # Drop rows with NaN values
                result_df = result_df.dropna()
                
                # Sort by date
                result_df = result_df.sort_values('date').reset_index(drop=True)
                
                if len(result_df) > 0:
                    st.success(f"✅ AI successfully extracted {len(result_df)} data rows for forecasting!")
                    
                    # Get column names for results
                    date_col_name = str(data_rows.columns[date_col])
                    value_col_name = str(data_rows.columns[value_col])
                    
                    st.write(f"Using '{date_col_name}' for dates and '{value_col_name}' for values")
                    st.write("Extracted Data:")
                    st.dataframe(result_df)
                    
                    return result_df, 'date', 'value'
        
        # AI ANALYSIS PHASE 4: No header row found, try direct numerical detection
        # -------------------------------------------------------------
        
        st.warning("⚠️ AI couldn't identify a clear header row, trying alternative analysis...")
        
        # Find columns with mostly numeric values (potential price columns)
        numeric_cols = []
        for col_idx in range(df_copy.shape[1]):
            numeric_count = 0
            for val in df_copy.iloc[:, col_idx]:
                if isinstance(val, (int, float)) and not pd.isna(val):
                    numeric_count += 1
                elif isinstance(val, str) and any(c.isdigit() for c in val):
                    numeric_count += 1
            
            if numeric_count > len(df_copy) / 4:  # At least 25% of values are numeric
                numeric_cols.append(col_idx)
        
        # Find columns with date-like values
        date_cols = []
        for col_idx in range(df_copy.shape[1]):
            date_count = 0
            for val in df_copy.iloc[:, col_idx]:
                if isinstance(val, str):
                    val_lower = val.lower()
                    if any(month in val_lower for month in 
                          ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                        date_count += 1
                    elif '/' in val or '-' in val or '.' in val:
                        parts = val.replace('/', ' ').replace('-', ' ').replace('.', ' ').split()
                        if all(part.isdigit() for part in parts) and 2 <= len(parts) <= 3:
                            date_count += 1
            
            if date_count > 0:
                date_cols.append((col_idx, date_count))
        
        # Sort date columns by count
        date_cols.sort(key=lambda x: x[1], reverse=True)
        
        # If we found date and numeric columns, try to use them
        if date_cols and numeric_cols:
            date_col = date_cols[0][0]  # Use the column with the most date-like values
            
            # Find the best numeric column (not the date column)
            value_col = None
            for col_idx in numeric_cols:
                if col_idx != date_col:
                    value_col = col_idx
                    break
            
            if value_col is not None:
                # Find rows with non-empty values in both columns
                valid_rows = []
                for idx in range(len(df_copy)):
                    date_val = df_copy.iloc[idx, date_col]
                    price_val = df_copy.iloc[idx, value_col]
                    
                    if not pd.isna(date_val) and not pd.isna(price_val):
                        # Skip rows with "Search Date" or other header indicators
                        row_text = ' '.join([str(val).lower() for val in df_copy.iloc[idx].values if isinstance(val, str)])
                        if 'search date' not in row_text and not any(val.lower() == 'date' for val in df_copy.iloc[idx].values if isinstance(val, str)):
                            valid_rows.append(idx)
                
                if valid_rows:
                    # Extract the valid rows
                    final_data = df_copy.iloc[valid_rows].copy()
                    
                    # Create the result dataframe
                    result_df = pd.DataFrame({
                        'date': pd.to_datetime(final_data.iloc[:, date_col], errors='coerce'),
                        'value': pd.to_numeric(
                            final_data.iloc[:, value_col].astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                            errors='coerce'
                        )
                    })
                    
                    # Drop rows with NaN values
                    result_df = result_df.dropna()
                    
                    # Sort by date
                    result_df = result_df.sort_values('date').reset_index(drop=True)
                    
                    if len(result_df) > 0:
                        st.success(f"✅ AI successfully extracted {len(result_df)} data rows using numerical detection!")
                        st.write("Extracted Data:")
                        st.dataframe(result_df)
                        
                        return result_df, 'date', 'value'
        
        # AI ANALYSIS PHASE 5: Try a scan of specific rows (starting from row 3)
        # -------------------------------------------------------------
        
        st.warning("⚠️ Trying comprehensive row scanning approach...")
        
        # Scan all rows starting from row 3
        headers_found = False
        header_row_idx = None
        num_columns = len(df_copy.columns)
        
        for idx in range(min(3, len(df_copy)), len(df_copy)):
            row = df_copy.iloc[idx]
            row_text = ' '.join([str(val).lower() for val in row.values if isinstance(val, str)])
            
            # If this row contains multiple header indicators, it might be the header row
            header_matches = sum(1 for indicator in header_indicators if indicator in row_text)
            
            if header_matches >= 2:
                header_row_idx = idx
                headers_found = True
                st.info(f"🔍 Found likely header row at index {idx}")
                break
        
        if headers_found and header_row_idx is not None:
            # Get everything after the header row
            data_start_idx = header_row_idx + 1
            if data_start_idx < len(df_copy):
                potential_data = df_copy.iloc[data_start_idx:].reset_index(drop=True)
                
                # Try to guess which columns contain dates and values
                date_col = num_columns - 1  # Last column often contains dates
                value_col = None
                
                # Check each column for numeric values
                for col_idx in range(num_columns):
                    if col_idx != date_col:
                        numeric_vals = pd.to_numeric(
                            potential_data.iloc[:, col_idx].astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                            errors='coerce'
                        ).notna().sum()
                        
                        if numeric_vals > 0:
                            value_col = col_idx
                            break
                
                if value_col is not None:
                    # Create the final dataframe
                    result_df = pd.DataFrame({
                        'date': pd.to_datetime(potential_data.iloc[:, date_col], errors='coerce'),
                        'value': pd.to_numeric(
                            potential_data.iloc[:, value_col].astype(str).str.replace(r'[^\d\.-]', '', regex=True),
                            errors='coerce'
                        )
                    })
                    
                    # Drop rows with NaN values
                    result_df = result_df.dropna()
                    
                    if len(result_df) > 0:
                        st.success(f"✅ Successfully extracted {len(result_df)} rows starting after header row!")
                        st.write("Extracted Data:")
                        st.dataframe(result_df)
                        
                        return result_df, 'date', 'value'
        
        # If all attempts failed, return empty dataframe
        st.error("❌ AI could not extract valid time series data from this table format.")
        return pd.DataFrame({'date': [], 'value': []}), 'date', 'value'

def extract_with_ai(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Extract time series data using the AI table extractor.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of:
        - Transformed DataFrame ready for forecasting
        - Date column name
        - Value column name
    """
    extractor = AITableExtractor()
    return extractor.extract_from_dataframe(df)
