"""
Zero-safe forecasting utility functions to handle division by zero errors in forecasting
Import these functions to safely perform forecasting operations without zero division errors
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('zero_safe_forecasting')

def safe_data_prep(data, target_col, min_value=1.0):
    """
    Aggressively prepare data for forecasting by handling zeros, negative values, and NaNs.
    This is an enhanced version that ensures no errors can occur during subsequent operations.
    
    Args:
        data: DataFrame with data to prepare
        target_col: Target column to prepare
        min_value: Minimum value to enforce in the data (will replace any lower values)
        
    Returns:
        Tuple of (DataFrame with prepared data, offset applied)
    """
    # Return early on invalid inputs
    if data is None or len(data) == 0:
        logger.warning("Empty data provided to safe_data_prep")
        return data, 0
    
    if target_col not in data.columns:
        logger.warning(f"Target column {target_col} not found in data")
        return data, 0
    
    # Make a deep copy to avoid modifying the original
    result = data.copy(deep=True)
    
    # Log input data stats
    logger.info(f"Safe data prep - rows: {len(result)}, NaNs: {result[target_col].isna().sum()}," 
              f" zeros: {(result[target_col] == 0).sum()}, neg: {(result[target_col] < 0).sum()}")
    
    # Handle infinity values first
    if np.isinf(result[target_col]).any():
        logger.warning(f"Replacing {np.isinf(result[target_col]).sum()} infinite values with NaN")
        result[target_col] = result[target_col].replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values with multiple methods to ensure everything is filled
    if result[target_col].isna().any():
        logger.info(f"Filling {result[target_col].isna().sum()} NaN values in {target_col}")
        # First try interpolation
        result[target_col] = result[target_col].interpolate(method='linear')
        
        # Then forward/backward fill
        result[target_col] = result[target_col].fillna(method='bfill').fillna(method='ffill')
        
        # If still have NaN values, try using median instead of mean (more robust to outliers)
        if result[target_col].isna().any():
            median_val = result[target_col].median()
            if np.isnan(median_val):
                median_val = min_value * 2  # Use a higher default for safety
            result[target_col] = result[target_col].fillna(median_val)
    
    # Handle zero or negative values with a more aggressive approach
    if (result[target_col] <= 0).any():
        zeros_count = (result[target_col] == 0).sum()
        neg_count = (result[target_col] < 0).sum()
        total_bad = zeros_count + neg_count
        logger.warning(f"Found {total_bad} problematic values: {zeros_count} zeros, {neg_count} negatives")
        
        # Use a much larger offset for safety
        min_val = result[target_col].min()
        
        # Calculate a safe offset - at least 100 to be extremely safe
        if min_val < 0:
            # For negative values, make them significantly positive
            offset = abs(min_val) + 100.0  # Very large safety margin
            logger.warning(f"Applying large offset of {offset} to handle negative values")
        else:
            # For zeros, apply a large offset as well
            offset = 100.0  # Much larger than traditional approaches
            logger.warning(f"Applying offset of {offset} to handle zeros")
        
        # Apply offset to all values
        result[target_col] = result[target_col] + offset
        
        # Verify the fix worked
        if (result[target_col] <= 0).any():
            # If we still have non-positive values, force them to min_value
            logger.error("Offset didn't fix all values, forcing minimum values")
            result[target_col] = result[target_col].clip(lower=min_value)
    else:
        offset = 0
        
    # Final safety check - any remaining problematic values are set to min_value
    if result[target_col].isna().any() or np.isinf(result[target_col]).any() or (result[target_col] <= 0).any():
        logger.error("Final check found problematic values - forcing safe values")
        result[target_col] = result[target_col].fillna(min_value)
        result[target_col] = result[target_col].replace([np.inf, -np.inf], min_value * 10)
        result[target_col] = result[target_col].clip(lower=min_value)
    
    logger.info(f"Data preparation complete with offset: {offset}")
    return result, offset

def safe_division(a, b, default=0.0, min_denominator=1e-10):
    """
    Ultra-safe division that handles all edge cases including zeros, NaNs, and infinities
    in both scalar and array-like inputs.
    
    Args:
        a: Numerator (scalar, numpy array, or pandas Series)
        b: Denominator (scalar, numpy array, or pandas Series)
        default: Default value to return when division is unsafe
        min_denominator: Minimum value to use for denominator to avoid division by zero
        
    Returns:
        Result of a/b, or default if division would be unsafe
    """
    logger.debug(f"Safe division called with types: a={type(a)}, b={type(b)}")
    
    # CASE 1: Both inputs are scalars
    if np.isscalar(a) and np.isscalar(b):
        # Handle special cases
        if np.isnan(a) or np.isnan(b) or np.isinf(a) or b == 0 or np.abs(b) < min_denominator:
            return default
        
        try:
            result = a / b
            # Check if result is valid
            if np.isnan(result) or np.isinf(result):
                return default
            return result
        except Exception:
            return default
    
    # CASE 2: Array-like inputs
    try:
        # Convert inputs to numpy arrays for consistent handling
        a_array = np.array(a, dtype=float) if not isinstance(a, np.ndarray) else a.astype(float)
        b_array = np.array(b, dtype=float) if not isinstance(b, np.ndarray) else b.astype(float)
        
        # Create a mask for safe division (non-zero, non-NaN, non-inf denominators)
        safe_mask = (np.abs(b_array) >= min_denominator) & ~np.isnan(b_array) & ~np.isinf(b_array) & ~np.isnan(a_array)
        
        # Create result array filled with default value
        result = np.full_like(b_array, default, dtype=float)
        
        # Replace with actual division only where safe
        result[safe_mask] = a_array[safe_mask] / b_array[safe_mask]
        
        # Replace any resulting NaN or inf with default
        result[~np.isfinite(result)] = default
        
        # Return as same type as input when possible
        if isinstance(b, pd.Series) and hasattr(b, 'index'):
            return pd.Series(result, index=b.index)
        elif isinstance(a, pd.Series) and hasattr(a, 'index'):
            return pd.Series(result, index=a.index)
        else:
            return result
    
    except Exception:
        return default

def safe_mape(actual, pred, epsilon=1.0):
    """
    Calculate Mean Absolute Percentage Error safely, handling zeros and negative values
    
    Args:
        actual: Actual values
        pred: Predicted values
        epsilon: Small value to add to prevent division by zero
        
    Returns:
        MAPE value (as percentage)
    """
    # Convert inputs to numpy arrays
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    
    # Replace zeros with epsilon
    safe_actual = np.maximum(np.abs(actual), epsilon)
    
    # Calculate MAPE
    return np.mean(np.abs((actual - pred) / safe_actual)) * 100

def safe_weights_calculation(weights_dict):
    """
    Safely normalize weights, handling zero or negative weights
    
    Args:
        weights_dict: Dictionary of item to weight mappings
        
    Returns:
        Dictionary with normalized weights
    """
    # Make a copy
    weights = weights_dict.copy()
    
    # Ensure all weights are positive
    for key in weights:
        if weights[key] <= 0:
            weights[key] = 0.1  # Set small positive weight
    
    # Normalize to sum to 1
    total = sum(weights.values())
    if total <= 0:
        # If total is zero or negative, use equal weights
        equal_weight = 1.0 / len(weights)
        return {key: equal_weight for key in weights}
    
    # Normal case - weights sum to a positive number
    return {key: weight/total for key, weight in weights.items()}

def safe_scale_data(data):
    """
    Safely scale data to mean 0 and standard deviation 1, handling zero variance
    
    Args:
        data: Data to scale
        
    Returns:
        Scaled data
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Ensure std is not zero to avoid division by zero
    if std <= 1e-10:
        std = 1.0
        
    return (data - mean) / std

# Easy-to-use wrappers for main forecasting functions
def fix_xgboost_division(data, target_col):
    """Prepare data specifically for XGBoost to avoid division by zero"""
    return safe_data_prep(data, target_col, min_value=10.0)  # Use higher min value for more stability

def fix_ensemble_weights(weights_dict):
    """Fix weights for ensemble forecasting to avoid division by zero"""
    return safe_weights_calculation(weights_dict)
