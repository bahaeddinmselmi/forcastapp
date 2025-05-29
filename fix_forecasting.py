"""Script to comprehensively fix forecasting.py issues"""

import re
import os

# Path to the forecasting.py file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\utils\forecasting.py"

# Create a backup of the original file
with open(file_path, 'r', encoding='utf-8') as f:
    original_content = f.read()

backup_path = file_path + '.fullbackup'
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(original_content)

print(f"Created backup at {backup_path}")

# Define the fixed ARIMA forecast function with better error handling
arima_fixed_code = """
def arima_forecast(train_data: pd.Series, 
                periods: int, 
                order: Tuple[int, int, int] = (1, 1, 1),
                seasonal_order: Tuple[int, int, int, int] = None,
                return_conf_int: bool = True,
                alpha: float = 0.05,
                future_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    \"\"\"
    Forecast using ARIMA or SARIMA model.
    
    Args:
        train_data: Training data as pandas Series with datetime index
        periods: Number of periods to forecast
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, S) for SARIMA
        return_conf_int: Whether to return confidence intervals
        alpha: Significance level for confidence intervals
        future_index: Optional custom future DatetimeIndex for forecast
        
    Returns:
        Dictionary with forecast results
    \"\"\"
    # Initialize for possible use in except blocks
    model = None
    forecast_series = None
    forecast_periods = periods
    
    try:
        # Suppress warnings during model fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            # Create and fit the model
            if seasonal_order is not None:
                model = SARIMAX(
                    train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(
                    train_data, 
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            model_fit = model.fit(disp=0)  # Fit without displaying convergence info
            
            # Generate forecast based on whether custom dates are provided
            if future_index is not None:
                forecast_periods = len(future_index)
                forecast_values = model_fit.forecast(steps=forecast_periods)
                forecast_series = pd.Series(forecast_values, index=future_index)
            else:
                forecast_series = model_fit.forecast(steps=periods)
            
            # Prepare the result dictionary
            result = {
                "model": "SARIMA" if seasonal_order else "ARIMA",
                "forecast": forecast_series,
                "params": {
                    "order": order,
                    "seasonal_order": seasonal_order
                }
            }
            
            # Add confidence intervals if requested
            if return_conf_int:
                # Get confidence intervals with a try-except to handle potential errors
                try:
                    if future_index is not None:
                        # For custom index, we need to get prediction intervals differently
                        pred_intervals = model_fit.get_forecast(steps=forecast_periods).conf_int(alpha=alpha)
                        lower = pd.Series(pred_intervals.iloc[:, 0].values, index=future_index)
                        upper = pd.Series(pred_intervals.iloc[:, 1].values, index=future_index)
                    else:
                        # For standard forecast
                        pred_intervals = model_fit.get_forecast(steps=periods).conf_int(alpha=alpha)
                        lower = pred_intervals.iloc[:, 0]
                        upper = pred_intervals.iloc[:, 1]
                    
                    result["lower"] = lower
                    result["upper"] = upper
                except Exception as e_conf:
                    # If confidence intervals fail, continue without them
                    print(f"Warning: Could not generate confidence intervals: {str(e_conf)}")
                    result["conf_int_error"] = str(e_conf)
            
            return result
            
    except Exception as e:
        print(f"ARIMA model fitting failed: {str(e)}")
        
        try:
            # Try a simpler model (1,0,0) without seasonality as fallback
            print("Trying simpler ARIMA model as fallback...")
            
            simple_model = ARIMA(train_data, order=(1,0,0))
            simple_fit = simple_model.fit(disp=0)
            
            # Generate forecast
            if future_index is not None:
                forecast_periods = len(future_index)
                forecast_values = simple_fit.forecast(steps=forecast_periods)
                forecast_series = pd.Series(forecast_values, index=future_index)
            else:
                forecast_series = simple_fit.forecast(steps=periods)
                
            return {
                "model": "Simple ARIMA(1,0,0) - Fallback",
                "forecast": forecast_series,
                "error": str(e),
                "note": "Using simpler ARIMA model due to original model failure"
            }
            
        except Exception as e2:
            print(f"Simple ARIMA fallback also failed: {str(e2)}")
            
            # Last resort: naive forecast (use last value)
            try:
                last_value = train_data.iloc[-1]
                if future_index is not None:
                    pred_series = pd.Series([last_value] * len(future_index), index=future_index)
                else:
                    pred_series = pd.Series([last_value] * periods)
                
                return {
                    'forecast': pred_series,
                    'model': 'Naive Forecast',
                    'note': 'Using last-value forecast due to ARIMA model failures'
                }
            except Exception as e3:
                # Complete failure - return an empty forecast with error info
                print(f"All forecast attempts failed: {str(e3)}")
                return {
                    'error': f"ARIMA model failed: {str(e)}. All backups also failed.",
                    'model': 'Failed Forecast'
                }
"""

# Fix the content - first add warning suppression
warning_imports = """
# Add warning suppression
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence common warnings that affect forecasting
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")
warnings.filterwarnings("ignore", message="Covariance matrix is singular")
"""

# Read the current content to find patterns to replace
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add warning imports after the Prophet import
content_with_warnings = re.sub(
    r'from prophet import Prophet(\s*)',
    f'from prophet import Prophet\\1{warning_imports}',
    content, 
    count=1
)

# Replace the arima_forecast function with our fixed version
# First find where the function starts
arima_pattern = r'def arima_forecast\([^)]*\)[^:]*:'
arima_match = re.search(arima_pattern, content_with_warnings)

if arima_match:
    # Find the function start
    func_start = arima_match.start()
    
    # Find where to end the replacement - look for the next function def that's at the same level
    # or the end of the file
    next_def_pattern = r'\ndef [a-zA-Z_]+'
    next_def_match = re.search(next_def_pattern, content_with_warnings[func_start:])
    
    if next_def_match:
        func_end = func_start + next_def_match.start()
        # Fixed content with arima function replaced
        fixed_content = content_with_warnings[:func_start] + arima_fixed_code + content_with_warnings[func_end:]
    else:
        # If no next function found, replace to the end (though this is unlikely)
        fixed_content = content_with_warnings[:func_start] + arima_fixed_code
        print("Warning: Couldn't find the end of the arima_forecast function. Replacement may be incomplete.")
else:
    # If we can't find the function, just add it at the end of the file
    fixed_content = content_with_warnings + "\n\n" + arima_fixed_code
    print("Warning: Couldn't find the arima_forecast function. Added it at the end of the file.")

# Save the fixed content
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Fixed forecasting.py - replaced arima_forecast function and added warning suppressions.")
print(f"Original file backed up to {backup_path}")
