"""
Forecasting utilities for the IBP system.
Implements various statistical and machine learning forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence common warnings from statsmodels
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def arima_forecast(train_data: pd.Series, 
                periods: int, 
                order: Tuple[int, int, int] = (1, 1, 1),
                seasonal_order: Tuple[int, int, int, int] = None,
                return_conf_int: bool = True,
                alpha: float = 0.05,
                future_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
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
    """
    try:
        # Temporarily suppress specific warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # Try to fit the model with the specified parameters
            if seasonal_order is not None:
                # Use SARIMA for seasonal data
                model = SARIMAX(
                    train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                # Use ARIMA for non-seasonal data
                model = ARIMA(
                    train_data, 
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            # Fit the model with relaxed convergence criteria
            model_fit = model.fit(method="powell", disp=False, maxiter=100)
            
            # Determine forecast periods based on future_index if provided
            if future_index is not None:
                forecast_periods = len(future_index)
            else:
                forecast_periods = periods
            
            # Generate forecast with or without confidence intervals
            if return_conf_int:
                forecast = model_fit.get_forecast(steps=forecast_periods)
                forecast_mean = forecast.predicted_mean
                forecast_ci = forecast.conf_int(alpha=alpha)
                
                if future_index is not None:
                    # Map predictions to custom future index
                    pred_series = pd.Series(forecast_mean.values, index=future_index)
                    lower_series = pd.Series(forecast_ci.iloc[:, 0].values, index=future_index)
                    upper_series = pd.Series(forecast_ci.iloc[:, 1].values, index=future_index)
                else:
                    # Use the model's generated index
                    pred_series = forecast_mean
                    lower_series = forecast_ci.iloc[:, 0]
                    upper_series = forecast_ci.iloc[:, 1]
            else:
                # Generate forecast without confidence intervals
                forecast_mean = model_fit.forecast(steps=forecast_periods)
                if future_index is not None:
                    pred_series = pd.Series(forecast_mean.values, index=future_index)
                else:
                    pred_series = forecast_mean
                lower_series = upper_series = None
        
        # Return results
        results = {
            "forecast": pred_series,
            "model": "ARIMA" if seasonal_order is None else "SARIMA",
            "order": order,
            "seasonal_order": seasonal_order
        }
        
        # Add confidence intervals if available
        if return_conf_int and lower_series is not None and upper_series is not None:
            results["lower_bound"] = lower_series
            results["upper_bound"] = upper_series
            
        return results
        
    except Exception as e:
        print(f"Error in ARIMA forecast: {str(e)}. Trying simplified model...")
        
        # If first attempt failed, try a simpler model with more robust parameters
        try:
            # Use simpler model parameters that are more likely to converge
            simple_order = (1, 1, 0)  # Simple AR(1) with differencing
            simple_seasonal = (1, 0, 0, 12) if seasonal_order else None  # Simple seasonal if needed
            
            if simple_seasonal:
                model = SARIMAX(
                    train_data,
                    order=simple_order,
                    seasonal_order=simple_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(
                    train_data, 
                    order=simple_order
                )
            
            # Fit with more robust method
            model_fit = model.fit(method="powell", disp=False)
            
            # Determine forecast periods
            if future_index is not None:
                forecast_periods = len(future_index)
            else:
                forecast_periods = periods
            
            # Simple forecast without confidence intervals to avoid errors
            forecast_mean = model_fit.forecast(steps=forecast_periods)
            
            if future_index is not None:
                pred_series = pd.Series(forecast_mean.values, index=future_index)
            else:
                pred_series = forecast_mean
            
            # Return simplified results
            return {
                "forecast": pred_series,
                "model": "Simple ARIMA",
                "order": simple_order,
                "seasonal_order": simple_seasonal,
                "note": "Using simplified model due to convergence issues"
            }
            
        except Exception as e2:
            # If even the simple model fails, fall back to naive forecast
            print(f"Simple ARIMA failed: {str(e2)}. Using naive forecast.")
            try:
                # Try to create a reasonable empty index
                if future_index is not None:
                    index_to_use = future_index
                else:
                    # Create an index based on last_date + periods
                    try:
                        last_date = train_data.index[-1]
                        freq = pd.infer_freq(train_data.index) if train_data.index.freq is None else train_data.index.freq
                        index_to_use = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
                    except Exception:
                        # Last resort: create a default numeric index
                        index_to_use = range(periods)
                        
                # Use a super simple last-value forecast
                last_value = train_data.iloc[-1]
                forecast_values = [last_value] * periods
                    
                pred_series = pd.Series(forecast_values, index=index_to_use)
                
                return {
                    "forecast": pred_series,
                    "model": "Naive Forecast",
                    "note": "Using last-value forecast due to ARIMA model failures"
                }
            except Exception as e3:
                # Complete failure - return an empty forecast with error info
                print(f"All forecast attempts failed: {str(e3)}")
                return {
                    "error": f"ARIMA model failed: {str(e)}. All backups also failed.",
                    "model": "Failed Forecast"
                }
