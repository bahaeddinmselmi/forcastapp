"""
Model Enhancement Module

This module provides helper functions to enhance forecasting models
with actual data feedback.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
from typing import Dict, Any, Tuple

def save_training_data(train_data: pd.Series, target_col: str):
    """
    Save the original training data to session state for later enhancement
    
    Args:
        train_data: The original training data
        target_col: The target column name
    """
    # Store it in session state
    st.session_state.original_training_data = train_data
    st.session_state.target_column = target_col
    
    # Also create a history directory if it doesn't exist
    os.makedirs("forecast_history", exist_ok=True)

def get_enhanced_training_data(train_data: pd.DataFrame, target_col: str) -> Tuple[pd.Series, bool]:
    """
    Get the training data, using enhanced data if available
    
    Args:
        train_data: The original input data
        target_col: The target column name
        
    Returns:
        Tuple of (training_data, is_enhanced)
    """
    # Check if enhanced data is available
    if 'use_enhanced_data' in st.session_state and st.session_state.use_enhanced_data and 'enhanced_training_data' in st.session_state:
        # Use enhanced data from actuals feedback
        return st.session_state.enhanced_training_data, True
    else:
        # If no enhanced data, extract target from original data
        if isinstance(train_data, pd.Series):
            return train_data, False
        elif isinstance(train_data, pd.DataFrame) and target_col in train_data.columns:
            return train_data[target_col], False
        else:
            # Last resort - try first column if it's a dataframe
            if isinstance(train_data, pd.DataFrame) and not train_data.empty:
                return train_data.iloc[:, 0], False
            else:
                raise ValueError("Cannot extract training data from the provided input")

def display_enhancement_status():
    """Display a message if enhanced data is being used"""
    if 'use_enhanced_data' in st.session_state and st.session_state.use_enhanced_data and 'enhanced_training_data' in st.session_state:
        st.success("✅ Using enhanced training data with actual results for better forecasts!")
        
        # Show details about the enhancement
        if 'enhanced_training_data' in st.session_state:
            original_len = len(st.session_state.get('original_training_data', pd.Series()))
            enhanced_len = len(st.session_state.enhanced_training_data)
            
            if original_len > 0:
                new_points = enhanced_len - original_len
                if new_points > 0:
                    st.info(f"Enhanced with {new_points} additional data points from actual values.")
