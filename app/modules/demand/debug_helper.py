"""
Debug helper functions to trace and fix data type issues
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Any

def safe_check_type(data: Any, name: str = "data") -> None:
    """
    Safely check an object's type and display helpful information
    
    Args:
        data: Any Python object to check
        name: Name to display in the debug output
    """
    if data is None:
        st.info(f"DEBUG: {name} is None")
        return
        
    data_type = type(data).__name__
    st.info(f"DEBUG: {name} is type {data_type}")
    
    if isinstance(data, pd.DataFrame):
        st.info(f"DataFrame columns: {list(data.columns)}")
        st.info(f"DataFrame shape: {data.shape}")
    elif isinstance(data, pd.Series):
        st.info(f"Series length: {len(data)}")
        st.info(f"Series name: {data.name}")
        
def ensure_dataframe(data: Any, target_col: str = "value") -> pd.DataFrame:
    """
    Ensure data is a DataFrame, converting if needed
    
    Args:
        data: DataFrame or Series
        target_col: Column name to use if converting Series to DataFrame
    
    Returns:
        DataFrame object
    """
    if data is None:
        return pd.DataFrame()
        
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame
        df = pd.DataFrame({target_col: data})
        df.index = data.index
        return df
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        # Try to convert other objects to DataFrame
        try:
            return pd.DataFrame(data)
        except:
            st.error(f"Could not convert {type(data).__name__} to DataFrame")
            return pd.DataFrame()
