"""
Utility functions for date handling in the IBP application
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import streamlit as st

def parse_date_formats(value, selected_year=None, default_day=1, force_year=False):
    """
    Enhanced date parser that handles many different date formats
    including month names, numeric formats, and special cases.
    
    Args:
        value: The input value to parse
        selected_year: Year to use for dates missing year information
        default_day: Default day to use when only month and year are provided
        force_year: Whether to force the selected_year for all dates
        
    Returns:
        Parsed datetime or NaT if parsing fails
    """
    # Use current year as default if selected_year is None
    if selected_year is None:
        selected_year = datetime.now().year
        
    if pd.isna(value):
        return pd.NaT
    
    # Handle Excel serial dates
    if isinstance(value, (int, float)) and not pd.isna(value):
        try:
            # Convert Excel serial date to datetime
            return pd.Timestamp.fromordinal(int(value) + 693594)
        except:
            pass
    
    # Convert to string for parsing text formats
    try:
        if not isinstance(value, str):
            value = str(value)
        
        # Clean up the string
        value = value.strip().lower()
        
        # PRIORITY CASE #1: Direct handling for "Mar-20" format (highest priority)
        # This is the most problematic case, so handle it specially and early
        month_abbr_pattern = re.compile(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\-\s]?(\d{2})$', re.IGNORECASE)
        month_match = month_abbr_pattern.match(value)
        if month_match:
            month_abbr = month_match.group(1).lower()
            two_digit_year = month_match.group(2)
            
            # Month mapping
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            if month_abbr in month_map:
                month_num = month_map[month_abbr]
                # Use 2000s as default for two-digit years unless > 50
                year_prefix = 1900 if int(two_digit_year) > 50 else 2000
                original_year = year_prefix + int(two_digit_year)
                
                # Use selected_year if forcing, otherwise use computed year
                use_year = selected_year if force_year else original_year
                
                # Create timestamp using datetime first, then convert to pandas timestamp
                # This approach is more reliable than direct pd.Timestamp creation
                try:
                    dt = datetime(year=use_year, month=month_num, day=1)
                    return pd.Timestamp(dt)
                except Exception as e:
                    print(f"Error with direct timestamp creation for {month_abbr}-{two_digit_year}, trying fallback: {e}")
                    # Fallback - use the current year
                    dt = datetime(year=datetime.now().year, month=month_num, day=1)
                    return pd.Timestamp(dt)
        
        # PRIORITY CASE #2: Other month name formats
        # Month name mappings for various languages and formats
        month_map = {
            'jan': 1, 'january': 1, 'enero': 1, 
            'feb': 2, 'february': 2, 'febrero': 2, 
            'mar': 3, 'march': 3, 'marzo': 3,
            'apr': 4, 'april': 4, 'abril': 4, 
            'may': 5, 'mayo': 5,
            'jun': 6, 'june': 6, 'junio': 6, 
            'jul': 7, 'july': 7, 'julio': 7,
            'aug': 8, 'august': 8, 'agosto': 8, 
            'sep': 9, 'sept': 9, 'september': 9, 'septiembre': 9,
            'oct': 10, 'october': 10, 'octubre': 10,
            'nov': 11, 'november': 11, 'noviembre': 11, 
            'dec': 12, 'december': 12, 'diciembre': 12,
            'q1': 3, 'quarter 1': 3, 'trimestre 1': 3,
            'q2': 6, 'quarter 2': 6, 'trimestre 2': 6,
            'q3': 9, 'quarter 3': 9, 'trimestre 3': 9,
            'q4': 12, 'quarter 4': 12, 'trimestre 4': 12
        }
        
        # SAFE APPROACH: Try direct string pattern identification
        # This is more reliable than pd.to_datetime for problematic formats
        for month_name, month_num in month_map.items():
            if month_name in value.lower():
                # Find year in the string if present
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', value)
                year = int(year_match.group(1)) if year_match else selected_year
                
                # Apply force_year if requested
                if force_year:
                    year = selected_year
                    
                try:
                    # Create using datetime first for safety
                    dt = datetime(year=year, month=month_num, day=default_day)
                    return pd.Timestamp(dt)
                except Exception as e:
                    print(f"Error creating date with {month_name} and year {year}: {e}")
                    # Fallback to current year if there's an error
                    try:
                        dt = datetime(year=datetime.now().year, month=month_num, day=default_day)
                        return pd.Timestamp(dt)
                    except:
                        pass
        
        # FALLBACK: Try with pandas' flexible parser
        try:
            parsed_date = pd.to_datetime(value, errors='coerce')
            if pd.notna(parsed_date):
                # Apply force_year if needed
                if force_year and selected_year is not None:
                    try:
                        return parsed_date.replace(year=selected_year)
                    except:
                        # Some timestamps can't have their year changed to certain values
                        # In that case, return the original
                        return parsed_date
                return parsed_date
        except Exception as e:
            print(f"Error with pandas date parsing for {value}: {e}")
            pass
        
        # Check for month names (Jan, February, etc.)
        for month_str, month_num in month_map.items():
            if month_str in value:
                # Extract year from the string if possible
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', value)
                year = int(year_match.group(1)) if year_match else selected_year
                
                # Try to extract a day number
                day_match = re.search(r'\b(3[01]|[12]\d|0?[1-9])(?!\d)', value)
                day = int(day_match.group(1)) if day_match else default_day
                
                try:
                    return pd.Timestamp(year=year, month=month_num, day=day)
                except:
                    # If invalid day (e.g., Feb 30), use the last day of the month
                    try:
                        if day > 28:
                            return pd.Timestamp(year=year, month=month_num, day=1) + pd.offsets.MonthEnd(0)
                    except:
                        pass
        
        # Check for MM/DD/YYYY, DD/MM/YYYY, YYYY/MM/DD formats with various separators
        date_patterns = [
            # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
            r'(19|20)\d{2}[\-/\.](0?[1-9]|1[0-2])[\-/\.](0?[1-9]|[12]\d|3[01])',
            # MM-DD-YYYY or MM/DD/YYYY or MM.DD.YYYY
            r'(0?[1-9]|1[0-2])[\-/\.](0?[1-9]|[12]\d|3[01])[\-/\.](19|20)\d{2}',
            # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
            r'(0?[1-9]|[12]\d|3[01])[\-/\.](0?[1-9]|1[0-2])[\-/\.](19|20)\d{2}',
            # YYYY-MM or YYYY/MM or YYYY.MM
            r'(19|20)\d{2}[\-/\.](0?[1-9]|1[0-2])'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, value)
            if match:
                try:
                    # Try to parse with pandas
                    date_str = match.group(0)
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                    if pd.notna(parsed_date):
                        return parsed_date
                except:
                    pass
        
        # Try parsing month numbers directly (like "Month: 3")
        month_num_match = re.search(r'\b(1[0-2]|0?[1-9])\b', value)
        if month_num_match:
            try:
                month_num = int(month_num_match.group(1))
                if 1 <= month_num <= 12:
                    # Look for year in the string
                    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', value)
                    year = int(year_match.group(1)) if year_match else selected_year
                    return pd.Timestamp(year=year, month=month_num, day=default_day)
            except:
                pass
        
        # FINAL FALLBACK: Custom direct pattern matching
        try:
            # Check if there's just a plain integer representing a month
            if str(value).isdigit() and 1 <= int(value) <= 12:
                month_num = int(value)
                year = selected_year or datetime.now().year
                return pd.Timestamp(year=year, month=month_num, day=default_day)
                
            # Try to extract any month number from the string
            month_num_match = re.search(r'\b(1[0-2]|0?[1-9])\b', str(value))
            if month_num_match:
                try:
                    month_num = int(month_num_match.group(1))
                    if 1 <= month_num <= 12:
                        # Try to find a year in the string
                        year_pattern = re.search(r'\b(19\d{2}|20\d{2})\b', str(value))
                        year = int(year_pattern.group(1)) if year_pattern else (selected_year or datetime.now().year)
                        
                        # Apply force_year if needed
                        if force_year and selected_year is not None:
                            year = selected_year
                            
                        # Use datetime first for better error handling
                        dt = datetime(year=year, month=month_num, day=default_day)
                        return pd.Timestamp(dt)
                except Exception as e:
                    print(f"Error in final fallback parsing for {value}: {e}")
                    # Try one last approach with current date as baseline
                    try:
                        today = datetime.now()
                        dt = datetime(year=today.year, month=month_num, day=default_day)
                        return pd.Timestamp(dt)
                    except:
                        pass
        except Exception as e:
            print(f"Error in fallback date parsing for {value}: {e}")
    
    except Exception as e:
        print(f"Error in overall date parsing for '{value}': {str(e)}")
    
    # If absolutely all parsing attempts fail
    print(f"WARNING: Could not parse date value: {value} - returning NaT")
    return pd.NaT

def create_future_date_range(last_date, periods, freq='MS'):
    """
    Create a future date range starting from the month after the last date
    
    Args:
        last_date: The last date in the historical data
        periods: Number of periods to forecast
        freq: Frequency of the date range
        
    Returns:
        DatetimeIndex for future dates
    """
    try:
        # If last_date is NaT or None, use current date as fallback
        if pd.isna(last_date) or last_date is None:
            print(f"Warning: Invalid last_date {last_date}, using current date as fallback")
            last_date = pd.Timestamp.now()
            
        # Convert to timestamp if not already
        if not isinstance(last_date, pd.Timestamp):
            try:
                last_date = pd.Timestamp(last_date)
            except Exception as e:
                print(f"Error converting {last_date} to timestamp: {e}")
                # Fallback to current date if conversion fails
                last_date = pd.Timestamp.now()
            
        # Start the forecast from the last date
        forecast_start_date = last_date
        print(f"Creating forecast starting from {forecast_start_date}")
        
        # Create a proper future index
        future_index = pd.date_range(
            start=forecast_start_date,
            periods=periods,
            freq=freq
        )
        
        return future_index
    except Exception as e:
        st.error(f"Error creating future date range: {str(e)}")
        # Fallback to a generic future range
        return pd.date_range(start=datetime.now(), periods=periods, freq=freq)
