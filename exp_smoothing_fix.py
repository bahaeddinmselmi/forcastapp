"""Script to add the missing exp_smoothing_forecast function to forecasting.py"""

import os

# The implementation of exp_smoothing_forecast to add
exp_smoothing_code = """
def exp_smoothing_forecast(train_data: pd.Series, 
                        periods: int,
                        seasonal_periods: int = None,
                        trend: str = 'add',
                        seasonal: str = 'add',
                        damped_trend: bool = False,
                        future_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    \"\"\"
    Forecast using Exponential Smoothing model.
    
    Args:
        train_data: Training data as pandas Series with datetime index
        periods: Number of periods to forecast
        seasonal_periods: Number of periods in a season
        trend: Type of trend component ('add', 'mul', None)
        seasonal: Type of seasonal component ('add', 'mul', None)
        damped_trend: Whether to use damped trend
        future_index: Optional custom future DatetimeIndex for forecast
        
    Returns:
        Dictionary with forecast results
    \"\"\"
    try:
        # Determine if we should use seasonal component
        use_seasonal = seasonal is not None and seasonal_periods is not None
        
        # Create and fit the model
        if use_seasonal:
            model = ExponentialSmoothing(
                train_data,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            )
        else:
            # Non-seasonal model
            model = ExponentialSmoothing(
                train_data,
                trend=trend,
                seasonal=None,
                damped_trend=damped_trend
            )
            
        model_fit = model.fit(optimized=True)
        
        # Generate forecast with custom dates if provided
        if future_index is not None:
            # User wants to forecast for specific future dates
            forecast_periods = len(future_index)
            custom_dates_info = f"Using custom date range starting from {future_index[0].strftime('%Y-%m-%d')}"
            
            # Get forecast and map to custom dates
            forecast_values = model_fit.forecast(steps=forecast_periods)
            forecast_series = pd.Series(forecast_values, index=future_index)
        else:
            # Standard forecast
            forecast_series = model_fit.forecast(steps=periods)
        
        # Prepare result
        result = {
            "model": "Exponential Smoothing",
            "forecast": forecast_series
        }
        
        # Add info about custom dates if applicable
        if future_index is not None:
            result["custom_dates_info"] = custom_dates_info
            
        return result
        
    except Exception as e:
        # Return empty forecast with error message
        print(f"Error in Exponential Smoothing forecast: {str(e)}")
        
        # Try a simple moving average as fallback
        try:
            print("Trying simple moving average as fallback...")
            # Use last value as simple forecast
            last_value = train_data.iloc[-1]
            
            if future_index is not None:
                forecast = pd.Series([last_value] * len(future_index), index=future_index)
            else:
                forecast = pd.Series([last_value] * periods)
                
            return {
                "model": "Simple Average (Fallback)",
                "forecast": forecast,
                "error": str(e),
                "note": "Using simple average due to Exponential Smoothing failure"
            }
        except Exception as e2:
            return {
                "error": f"Exponential Smoothing forecast failed: {str(e)}. Fallback also failed: {str(e2)}",
                "model": "Failed Forecast"
            }
"""

# Path to the forecasting.py file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\utils\forecasting.py"

# Add the exp_smoothing_forecast function to the end of the file
with open(file_path, 'a') as f:
    f.write("\n\n# Added exp_smoothing_forecast function\n")
    f.write(exp_smoothing_code)

print("Added exp_smoothing_forecast function to forecasting.py")
