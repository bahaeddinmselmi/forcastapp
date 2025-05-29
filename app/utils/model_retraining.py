"""
Model Retraining Module

This module provides functionality to retrain forecasting models 
using feedback from actual data to improve future forecasts.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
from typing import Dict, Any, List, Optional

def save_actuals_to_history(actuals: pd.Series, history_dir: str = "forecast_history"):
    """
    Save actual values to the forecast history directory
    
    Args:
        actuals: Series of actual values with datetime index
        history_dir: Directory to save history data
    """
    # Create directory if it doesn't exist
    os.makedirs(history_dir, exist_ok=True)
    
    # Save actuals with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(history_dir, f"actuals_{timestamp}.pkl")
    
    # Convert actuals to DataFrame for easier loading later
    df = pd.DataFrame({"value": actuals})
    
    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)
    
    # Also save a CSV version for human readability
    csv_path = os.path.join(history_dir, f"actuals_{timestamp}.csv")
    df.to_csv(csv_path)
    
    return filepath

def get_historical_data(history_dir: str = "forecast_history"):
    """
    Load all historical actual data
    
    Args:
        history_dir: Directory with history data
        
    Returns:
        DataFrame of all historical actuals
    """
    # Create directory if it doesn't exist
    os.makedirs(history_dir, exist_ok=True)
    
    # Find all pickle files that start with "actuals_"
    all_data = None
    
    for filename in os.listdir(history_dir):
        if filename.startswith("actuals_") and filename.endswith(".pkl"):
            filepath = os.path.join(history_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    df = pickle.load(f)
                
                if all_data is None:
                    all_data = df
                else:
                    all_data = pd.concat([all_data, df])
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
    
    # If no data was found, return empty DataFrame
    if all_data is None:
        return pd.DataFrame()
    
    # Sort by date
    all_data = all_data.sort_index()
    
    return all_data

def combine_with_existing_data(new_actuals: pd.Series, train_data: pd.Series) -> pd.Series:
    """
    Combine new actual values with existing training data
    
    Args:
        new_actuals: Series of new actual values
        train_data: Series of existing training data
        
    Returns:
        Combined Series with both datasets
    """
    # Convert to DataFrames for easier manipulation
    new_df = pd.DataFrame({"value": new_actuals})
    train_df = pd.DataFrame({"value": train_data})
    
    # Combine and handle overlaps (new data takes precedence)
    combined = pd.concat([train_df, new_df])
    
    # Remove duplicates, keeping the latest entry
    combined = combined[~combined.index.duplicated(keep='last')]
    
    # Sort by date
    combined = combined.sort_index()
    
    # Convert back to Series
    return combined["value"]

def get_enhanced_train_data(actuals: pd.Series, original_train_data: pd.Series) -> pd.Series:
    """
    Get enhanced training data by combining original data with new actuals
    
    Args:
        actuals: Series of actual values
        original_train_data: Original training data
        
    Returns:
        Enhanced training data
    """
    # Save actuals to history
    save_actuals_to_history(actuals)
    
    # Get all historical data
    historical = get_historical_data()
    
    if historical.empty:
        # If no historical data was loaded, just use the current actuals
        historical_series = actuals
    else:
        # Convert to Series
        historical_series = historical["value"]
    
    # Combine with original training data
    enhanced_data = combine_with_existing_data(historical_series, original_train_data)
    
    return enhanced_data
