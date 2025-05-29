"""
AI-based Excel table detector.
Automatically identifies the location of data tables within Excel files.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import re

def detect_data_table(excel_file, sheet_name=None) -> Dict[str, Any]:
    """
    Automatically detect data table locations in an Excel file 
    using pattern recognition and heuristics.
    
    Args:
        excel_file: File object or path to the Excel file
        sheet_name: Optional sheet name to analyze; if None, will check all sheets
        
    Returns:
        Dict containing the detected table parameters:
            - sheet_name: The sheet containing the data
            - skiprows: Number of rows to skip
            - header: Header row position or None
            - usecols: Columns to use or None for all
    """
    excel = pd.ExcelFile(excel_file)
    
    # If no sheet specified, check all sheets
    sheets_to_check = [sheet_name] if sheet_name else excel.sheet_names
    best_result = None
    best_score = -1
    
    for sheet in sheets_to_check:
        # Read the entire sheet to analyze
        df_raw = pd.read_excel(excel, sheet_name=sheet, header=None)
        result = _analyze_sheet(df_raw)
        
        # If this sheet has a better score, keep it
        if result["score"] > best_score:
            best_score = result["score"]
            best_result = {
                "sheet_name": sheet,
                "skiprows": result["start_row"],
                "header": 0 if result["has_header"] else None,
                "usecols": result["col_range"],
                "score": result["score"],
            }
    
    # Return the best result found
    if best_result:
        # Remove score from final result
        del best_result["score"]
        return best_result
    
    # Fallback to default values if nothing detected
    return {
        "sheet_name": sheets_to_check[0],
        "skiprows": 0,
        "header": 0,
        "usecols": None
    }

def _analyze_sheet(df) -> Dict[str, Any]:
    """
    Analyze a sheet and find the most likely data table.
    
    Args:
        df: DataFrame containing raw sheet content
        
    Returns:
        Dict with detection results
    """
    rows, cols = df.shape
    
    # Initialize with defaults
    result = {
        "start_row": 0,
        "end_row": rows,
        "col_range": None,  # Use all columns
        "has_header": True,
        "score": 0,
    }
    
    # Skip initial empty rows
    first_non_empty = 0
    for i in range(rows):
        if not df.iloc[i].isna().all():
            first_non_empty = i
            break
    
    # Look for header patterns
    header_candidates = []
    
    # For each potential header row
    for i in range(first_non_empty, min(first_non_empty + 15, rows)):
        row = df.iloc[i]
        
        # Count non-null cells that could be headers
        non_null_count = (~row.isna()).sum()
        if non_null_count < 2:  # Need at least 2 columns for a proper table
            continue
            
        # Check if content looks like headers (text values, not values that look like data)
        header_score = _score_header_row(row, df, i)
        if header_score > 0:
            header_candidates.append((i, header_score))
    
    # If we found potential headers
    if header_candidates:
        # Sort by score descending
        header_candidates.sort(key=lambda x: x[1], reverse=True)
        potential_header_row = header_candidates[0][0]
        header_score = header_candidates[0][1]
        
        # Find the data range following this header
        data_rows, col_indices = _find_data_range(df, potential_header_row)
        
        if data_rows > 2:  # Need at least a few rows to be considered a table
            result["start_row"] = potential_header_row
            result["has_header"] = True
            result["end_row"] = potential_header_row + data_rows + 1  # +1 for header
            result["col_range"] = col_indices if col_indices else None
            result["score"] = header_score * data_rows  # Score based on header quality and data size
    else:
        # Try to find a data block without headers
        data_start, data_rows, col_indices = _find_data_block(df, first_non_empty)
        if data_rows > 5:  # Need more rows to be confident if no headers
            result["start_row"] = data_start
            result["has_header"] = False
            result["end_row"] = data_start + data_rows
            result["col_range"] = col_indices if col_indices else None
            result["score"] = data_rows  # Score based just on data size
    
    return result

def _score_header_row(row, df, row_idx) -> float:
    """
    Score a row based on how likely it is to be a header row.
    
    Higher score = more likely to be a header.
    """
    score = 0
    non_null_count = 0
    text_count = 0
    pattern_match = 0
    
    # Get next row for comparison
    next_row_idx = row_idx + 1
    if next_row_idx >= len(df):
        return 0  # Can't be a header if it's the last row
        
    next_row = df.iloc[next_row_idx]
    
    # Check each cell
    for i, cell in enumerate(row):
        # Skip null cells
        if pd.isna(cell):
            continue
            
        non_null_count += 1
        
        # Text cells are more likely to be headers
        if isinstance(cell, str):
            text_count += 1
            
            # Common header patterns: capitalized, short, no numbers
            if cell.strip():
                if cell.strip()[0].isupper():
                    score += 0.5
                if len(cell) < 30:  # Headers tend to be short
                    score += 0.5
                if not any(c.isdigit() for c in cell):
                    score += 0.5
                    
                # Check if it matches common header names
                common_headers = ["date", "time", "id", "name", "type", "value", "price", 
                                 "cost", "amount", "quantity", "total", "rate"]
                if any(keyword in cell.lower() for keyword in common_headers):
                    score += 2
        
        # Check for type difference between header and data
        # Headers are often text while data cells may be numeric
        if not pd.isna(next_row[i]):
            header_type = type(cell)
            data_type = type(next_row[i])
            
            if header_type != data_type:
                pattern_match += 1
    
    # Bonus for high percentage of text cells
    if non_null_count > 0:
        text_ratio = text_count / non_null_count
        score += text_ratio * 3
        
    # Bonus for high percentage of type differences
    if non_null_count > 0:
        pattern_ratio = pattern_match / non_null_count
        score += pattern_ratio * 2
    
    return score

def _find_data_range(df, header_row_idx) -> Tuple[int, List[int]]:
    """
    Find the range of data rows after a header row.
    
    Returns:
        Tuple of (number of data rows, list of column indices to use)
    """
    # Get header row to find active columns
    header_row = df.iloc[header_row_idx]
    
    # Find columns with headers
    active_cols = [i for i, cell in enumerate(header_row) if not pd.isna(cell)]
    
    if not active_cols:
        return 0, []
    
    # Check rows after the header
    data_rows = 0
    rows_total = len(df)
    consecutive_empty_rows = 0
    max_consecutive_empty = 3  # Allow up to 3 consecutive empty rows before ending
    
    for i in range(header_row_idx + 1, rows_total):
        row = df.iloc[i]
        
        # Count non-empty cells in active columns
        non_empty = sum(1 for col in active_cols if not pd.isna(row[col]))
        
        # If the row has data, count it and reset consecutive empty counter
        if non_empty > 0:
            consecutive_empty_rows = 0
            data_rows += 1
        else:
            # If row is empty, increment consecutive empty counter
            consecutive_empty_rows += 1
            
            # Only end if we've seen multiple empty rows in a row
            # AND we've already found some data rows
            if consecutive_empty_rows >= max_consecutive_empty and data_rows > 0:
                break
            
            # Still count this row as part of the data region unless at the very beginning
            if data_rows > 0:
                data_rows += 1
    
    return data_rows, active_cols

def _find_data_block(df, start_row) -> Tuple[int, int, List[int]]:
    """
    Find a block of consistent data without headers.
    
    Returns:
        Tuple of (start row, number of rows, list of column indices)
    """
    rows, cols = df.shape
    best_start = start_row
    best_rows = 0
    best_cols = []
    
    # Scan for blocks of consistent data
    for i in range(start_row, min(start_row + 30, rows - 5)):  # Check within first 30 rows after start
        # Look for consecutive rows with similar structure
        active_cols = [j for j in range(cols) if not pd.isna(df.iloc[i, j])]
        
        if len(active_cols) < 2:  # Need at least 2 columns
            continue
        
        # Check consecutive rows for similar structure
        consistent_rows = 0
        consecutive_empty = 0
        max_empty_allowed = 5  # Allow up to 5 empty rows before stopping
        
        for j in range(i, rows):
            row = df.iloc[j]
            col_match = sum(1 for col in active_cols if not pd.isna(row[col]))
            
            # If most active columns have data, count as consistent
            if col_match >= len(active_cols) * 0.5:  # Relaxed criteria: only 50% need to match
                consistent_rows += 1
                consecutive_empty = 0  # Reset empty counter
            else:
                consecutive_empty += 1
                # End if we have too many consecutive empty rows but have found data
                if consecutive_empty > max_empty_allowed and consistent_rows > 5:
                    break
                # Continue if we've already found some consistent data
                if consistent_rows > 0:
                    consistent_rows += 1
        
        # If we found a bigger block, update the result
        if consistent_rows > best_rows:
            best_start = i
            best_rows = consistent_rows
            best_cols = active_cols
    
    return best_start, best_rows, best_cols

def load_excel_with_smart_detection(file_path, sheet_name=None) -> pd.DataFrame:
    """
    Load an Excel file with automatic table detection.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Optional sheet name (will auto-detect if not provided)
        
    Returns:
        DataFrame with the detected data table
    """
    # First try the AI detection method
    table_params = detect_data_table(file_path, sheet_name)
    
    # Load the data with detected parameters
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=table_params["sheet_name"],
            header=table_params["header"],
            skiprows=table_params["skiprows"],
            usecols=table_params["usecols"]
        )
        
        # Check if we got a reasonable result (multiple rows)
        if len(df) >= 5:
            # Do some cleanup - remove completely empty rows
            df = df.dropna(how='all')
            return df
        
    except Exception as e:
        # If there was an error with the detected parameters, we'll try fallback methods
        pass
    
    # AI detection didn't work well, try a fallback approach
    excel = pd.ExcelFile(file_path)
    sheet_to_use = sheet_name if sheet_name else excel.sheet_names[0]
    
    # Fallback 1: Try to read with default parameters
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_to_use)
        if len(df) > 0:
            return df
    except:
        pass
    
    # Fallback 2: Try first row as header, skip first few rows
    for skip_rows in [0, 1, 2, 3, 4, 5]:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_to_use, skiprows=skip_rows)
            # If we get a non-empty dataframe with reasonable column headers, use it
            if len(df) > 0 and not df.columns.str.contains('Unnamed').all():
                return df
        except:
            continue
    
    # Fallback 3: Try to read all data without headers and check each potential header row
    try:
        # Read without assuming headers
        raw_data = pd.read_excel(file_path, sheet_name=sheet_to_use, header=None)
        
        # Find the first row with a good number of non-null values
        for i in range(min(10, len(raw_data))):
            if raw_data.iloc[i].notna().sum() >= 3:  # At least 3 columns with data
                # Try using this as the header row
                df = pd.read_excel(
                    file_path, 
                    sheet_name=sheet_to_use,
                    header=i,
                    skiprows=range(i)  # Skip rows before the header
                )
                if len(df) > 0:
                    return df
    except:
        pass
        
    # Last resort: just read the file with pandas defaults
    return pd.read_excel(file_path, sheet_name=sheet_to_use)
