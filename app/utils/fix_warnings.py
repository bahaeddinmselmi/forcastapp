"""Warning suppression utilities for forecasting models"""

import warnings
import pandas as pd
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def suppress_forecasting_warnings():
    """Suppress common warnings from forecasting libraries"""
    # Suppress common warnings from statsmodels and pandas
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Could not infer format")
    warnings.filterwarnings("ignore", message="Out of bounds nanosecond timestamp")
    warnings.filterwarnings("ignore", message="Optimization failed to converge")
    
    # Return True to confirm it ran
    return True

# Automatically run the function when this module is imported
suppress_forecasting_warnings()

def safely_handle_xgboost(func):
    """Decorator to safely handle XGBoost forecasting"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IndexError as e:
            print(f"Caught XGBoost index error: {str(e)}")
            # Return a simple fallback forecast
            if 'train_data' in kwargs:
                train_data = kwargs['train_data']
            else:
                train_data = args[0]  # Assume first arg is train_data
                
            if 'periods' in kwargs:
                periods = kwargs['periods']
            else:
                periods = args[1]  # Assume second arg is periods
                
            if 'future_index' in kwargs:
                future_index = kwargs['future_index']
            else:
                future_index = kwargs.get('future_index', None)
                
            # Create a simple fallback forecast
            last_value = train_data.iloc[-1] if hasattr(train_data, 'iloc') else np.mean(train_data)
            if future_index is not None:
                forecast = pd.Series([last_value] * len(future_index), index=future_index)
            else:
                forecast = pd.Series([last_value] * periods)
                
            return {
                'forecast': forecast,
                'model': 'Simple Fallback',
                'error': str(e)
            }
    return wrapperprint('Warning suppression module loaded successfully')
