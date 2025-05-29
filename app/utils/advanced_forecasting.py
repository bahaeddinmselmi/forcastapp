"""
Advanced forecasting models to complement the base models.
Includes neural networks, ensemble methods, and additional preprocessing techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from scipy import stats
import warnings
from datetime import datetime
import joblib
import os
from pathlib import Path
import tempfile

# Ensure TensorFlow logs are not too verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


def create_sequence_data(data: np.ndarray, 
                       n_steps_in: int, 
                       n_steps_out: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training time series models.
    
    Args:
        data: Input data array
        n_steps_in: Number of time steps for input
        n_steps_out: Number of time steps to predict
        
    Returns:
        Tuple of (X, y) with input sequences and target values
    """
    X, y = [], []
    
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out)])
    
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: Tuple[int, int], 
                   units: List[int] = [50, 50], 
                   dropout_rate: float = 0.2,
                   bidirectional: bool = False) -> tf.keras.Model:
    """
    Build an LSTM model for time series forecasting.
    
    Args:
        input_shape: Shape of input data (time steps, features)
        units: List of units for each LSTM layer
        dropout_rate: Dropout rate to prevent overfitting
        bidirectional: Whether to use bidirectional LSTM layers
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Add LSTM layers with dropout
    for i, unit in enumerate(units):
        # First layer needs input shape
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(LSTM(unit, return_sequences=(i < len(units) - 1), name=f'lstm_{i}'), 
                                        input_shape=input_shape, name=f'bidirectional_{i}'))
            else:
                model.add(LSTM(unit, return_sequences=(i < len(units) - 1),
                              input_shape=input_shape, name=f'lstm_{i}'))
        else:
            if bidirectional:
                model.add(Bidirectional(LSTM(unit, return_sequences=(i < len(units) - 1), name=f'lstm_{i}'), 
                                        name=f'bidirectional_{i}'))
            else:
                model.add(LSTM(unit, return_sequences=(i < len(units) - 1), name=f'lstm_{i}'))
        
        # Add dropout after each LSTM layer
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    return model


def lstm_forecast(train_data: pd.Series,
                 periods: int,
                 sequence_length: int = 12,
                 lstm_units: List[int] = [64, 32],
                 epochs: int = 50,
                 batch_size: int = 8,
                 future_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
    Forecast using LSTM neural network.
    
    Args:
        train_data: Training data as pandas Series with datetime index
        periods: Number of periods to forecast
        sequence_length: Number of time steps to use for input sequence
        lstm_units: List of units for each LSTM layer
        epochs: Number of training epochs
        batch_size: Batch size for training
        future_index: Optional index for forecast dates
        
    Returns:
        Dictionary with forecast results
    """
    # Check if GPU is available
    using_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    
    # Ensure we have enough data for the sequence
    if len(train_data) < sequence_length + 1:
        raise ValueError(f"Training data length ({len(train_data)}) must be greater than sequence length ({sequence_length})")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # Create sequences for training
    X, y = create_sequence_data(scaled_data, sequence_length, 1)
    y = y.reshape(y.shape[0], y.shape[2])  # Reshape to (samples, 1)
    
    # Build and train LSTM model
    model = build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        units=lstm_units
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0  # Set to 1 to see progress
    )
    
    # Prepare for forecasting
    forecast_values = np.zeros(periods)
    
    # Use the last sequence from training data as the initial input
    curr_seq = scaled_data[-sequence_length:].copy()
    
    # Generate forecast one step at a time
    for i in range(periods):
        # Reshape for model input
        curr_batch = curr_seq.reshape(1, sequence_length, 1)
        
        # Predict next value
        next_value = model.predict(curr_batch, verbose=0)[0][0]
        
        # Store the prediction
        forecast_values[i] = next_value
        
        # Update sequence by removing first value and adding the prediction
        curr_seq = np.append(curr_seq[1:], next_value)
    
    # Inverse transform to get actual values
    forecast_values = scaler.inverse_transform(forecast_values.reshape(-1, 1)).flatten()
    
    # Create forecast Series with correct index
    if future_index is None:
        # Generate future dates based on the frequency of the training data
        last_date = train_data.index[-1]
        
        # Try to infer frequency
        freq = pd.infer_freq(train_data.index)
        if freq is None:
            # If can't infer, try common frequencies
            for possible_freq in ['D', 'B', 'W', 'M', 'MS', 'Q', 'QS', 'A', 'AS']:
                try:
                    # Check if all dates align with this frequency
                    test_range = pd.date_range(start=train_data.index[0], 
                                             periods=len(train_data), 
                                             freq=possible_freq)
                    if all(d1.strftime('%Y-%m-%d') == d2.strftime('%Y-%m-%d') 
                          for d1, d2 in zip(test_range, train_data.index)):
                        freq = possible_freq
                        break
                except:
                    continue
                    
            if freq is None:
                # Default to monthly if can't detect
                freq = 'MS'
                print("Couldn't detect frequency, defaulting to monthly (MS)")
        
        # Create future dates
        future_index = pd.date_range(
            start=pd.date_range(start=last_date, periods=2, freq=freq)[1],
            periods=periods,
            freq=freq
        )
    
    # Create Series with the forecast values
    forecast_series = pd.Series(forecast_values, index=future_index)
    
    # Return dictionary with results
    return {
        'model': 'LSTM',
        'forecast': forecast_series,
        'model_object': model,
        'scaler': scaler,
        'training_history': history.history,
        'sequence_length': sequence_length,
        'using_gpu': using_gpu
    }


def auto_arima_forecast(train_data: pd.Series,
                      periods: int,
                      seasonal: bool = True,
                      max_order: Tuple[int, int, int] = (5, 2, 5),
                      max_seasonal_order: Tuple[int, int, int, int] = (2, 1, 2, 12),
                      return_conf_int: bool = True,
                      future_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
    Forecast using auto-ARIMA model selection.
    
    Args:
        train_data: Training data as pandas Series with datetime index
        periods: Number of periods to forecast
        seasonal: Whether to consider seasonal models
        max_order: Maximum (p,d,q) order to consider
        max_seasonal_order: Maximum seasonal (P,D,Q,s) order to consider
        return_conf_int: Whether to return confidence intervals
        future_index: Optional index for forecast dates
        
    Returns:
        Dictionary with forecast results
    """
    # Suppress statsmodels warnings
    warnings.filterwarnings("ignore")
    
    # Run auto_arima to find the best model
    model = pm.auto_arima(
        train_data,
        seasonal=seasonal,
        m=12 if seasonal else 1,  # Default to monthly seasonality
        max_p=max_order[0],
        max_d=max_order[1],
        max_q=max_order[2],
        max_P=max_seasonal_order[0] if seasonal else 0,
        max_D=max_seasonal_order[1] if seasonal else 0,
        max_Q=max_seasonal_order[2] if seasonal else 0,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        trace=False
    )
    
    # Generate forecast
    if return_conf_int:
        forecast, conf_int = model.predict(periods, return_conf_int=True, alpha=0.05)
    else:
        forecast = model.predict(periods)
        conf_int = None
    
    # Create future index if not provided
    if future_index is None:
        last_date = train_data.index[-1]
        freq = pd.infer_freq(train_data.index) or 'MS'  # Default to MS if can't detect
        
        future_index = pd.date_range(
            start=pd.date_range(start=last_date, periods=2, freq=freq)[1],
            periods=periods,
            freq=freq
        )
    
    # Create Series with the forecast values
    forecast_series = pd.Series(forecast, index=future_index)
    
    # Format and return results
    result = {
        'model': 'AutoARIMA',
        'forecast': forecast_series,
        'model_object': model,
        'model_order': model.order,
        'aic': model.aic()
    }
    
    if return_conf_int and conf_int is not None:
        result['lower_bound'] = pd.Series(conf_int[:, 0], index=future_index)
        result['upper_bound'] = pd.Series(conf_int[:, 1], index=future_index)
    
    return result


def ensemble_forecast(train_data: pd.Series,
                    periods: int,
                    models: List[str] = ['arima', 'es', 'lstm'],
                    weights: Optional[List[float]] = None,
                    future_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
    Create an ensemble forecast from multiple models.
    
    Args:
        train_data: Training data as pandas Series with datetime index
        periods: Number of periods to forecast
        models: List of models to include in ensemble
        weights: Optional weights for each model (if None, equal weighting)
        future_index: Optional index for forecast dates
        
    Returns:
        Dictionary with forecast results
    """
    from utils.forecasting import arima_forecast, exp_smoothing_forecast
    
    # Initialize storage for individual model forecasts
    model_forecasts = {}
    
    # Generate forecasts for each requested model
    for model_name in models:
        try:
            if model_name.lower() == 'arima':
                result = arima_forecast(
                    train_data=train_data,
                    periods=periods,
                    order=(2, 1, 2),
                    future_index=future_index
                )
                model_forecasts['ARIMA'] = result['forecast']
                
            elif model_name.lower() == 'es':
                result = exp_smoothing_forecast(
                    train_data=train_data,
                    periods=periods,
                    seasonal='add',
                    future_index=future_index
                )
                model_forecasts['ExponentialSmoothing'] = result['forecast']
                
            elif model_name.lower() == 'lstm':
                result = lstm_forecast(
                    train_data=train_data,
                    periods=periods,
                    future_index=future_index
                )
                model_forecasts['LSTM'] = result['forecast']
                
            elif model_name.lower() == 'auto_arima':
                result = auto_arima_forecast(
                    train_data=train_data,
                    periods=periods,
                    future_index=future_index
                )
                model_forecasts['AutoARIMA'] = result['forecast']
                
        except Exception as e:
            print(f"Error generating {model_name} forecast: {str(e)}")
    
    # Check if we have any successful forecasts
    if not model_forecasts:
        raise ValueError("No valid forecasts generated for ensemble")
    
    # Apply weights if provided, otherwise use equal weights
    if weights is None:
        weights = [1.0 / len(model_forecasts)] * len(model_forecasts)
    else:
        # Normalize weights to sum to 1
        weights = [w / sum(weights) for w in weights]
    
    # Create a list of all forecast Series
    all_forecasts = list(model_forecasts.values())
    
    # Check if indexes match
    first_index = all_forecasts[0].index
    for forecast in all_forecasts[1:]:
        if not forecast.index.equals(first_index):
            raise ValueError("Forecast indexes do not match, cannot create ensemble")
    
    # Create weighted ensemble
    ensemble_values = np.zeros(periods)
    for i, (model_name, forecast) in enumerate(model_forecasts.items()):
        ensemble_values += forecast.values * weights[i]
    
    # Create Series with the ensemble forecast
    ensemble_series = pd.Series(ensemble_values, index=first_index)
    
    # Return dictionary with results
    return {
        'model': 'Ensemble',
        'forecast': ensemble_series,
        'individual_forecasts': model_forecasts,
        'weights': dict(zip(model_forecasts.keys(), weights))
    }


def decompose_time_series(series: pd.Series, 
                        model: str = 'additive', 
                        period: Optional[int] = None) -> Dict[str, pd.Series]:
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Args:
        series: Input time series
        model: Decomposition model ('additive' or 'multiplicative')
        period: Seasonality period (if None, will attempt to infer)
        
    Returns:
        Dictionary with decomposition components
    """
    # Try to infer period if not provided
    if period is None:
        # Default periods for common frequencies
        freq = pd.infer_freq(series.index)
        if freq in ['D', 'B']:
            period = 7  # Weekly
        elif freq in ['M', 'MS']:
            period = 12  # Monthly
        elif freq in ['Q', 'QS']:
            period = 4  # Quarterly
        elif freq in ['A', 'AS']:
            period = 1  # Yearly
        else:
            # Default to 12 if can't determine
            period = 12
    
    # Perform decomposition
    result = seasonal_decompose(
        series,
        model=model,
        period=period
    )
    
    # Return components
    return {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.residual,
        'observed': result.observed,
        'period': period,
        'model': model
    }


def detect_anomalies(series: pd.Series, 
                  window: int = 12, 
                  sigma: float = 3.0) -> pd.Series:
    """
    Detect anomalies in a time series using moving average and standard deviation.
    
    Args:
        series: Input time series
        window: Rolling window size
        sigma: Number of standard deviations for threshold
        
    Returns:
        Series with boolean values indicating anomalies
    """
    # Calculate rolling mean and std
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Calculate upper and lower bounds
    upper_bound = rolling_mean + (rolling_std * sigma)
    lower_bound = rolling_mean - (rolling_std * sigma)
    
    # Identify anomalies
    anomalies = (series > upper_bound) | (series < lower_bound)
    
    return anomalies


def check_stationarity(series: pd.Series) -> Dict[str, Any]:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        series: Input time series
        
    Returns:
        Dictionary with test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Perform ADF test
    result = adfuller(series.dropna())
    
    # Get test statistics
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Determine if stationary
    is_stationary = p_value < 0.05
    
    return {
        'is_stationary': is_stationary,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values
    }


def calculate_forecast_trends(forecast: pd.Series) -> Dict[str, Any]:
    """
    Calculate trends and patterns in a forecast.
    
    Args:
        forecast: Forecast time series
        
    Returns:
        Dictionary with trend analysis
    """
    # Basic statistics
    stats = {
        'mean': forecast.mean(),
        'median': forecast.median(),
        'min': forecast.min(),
        'max': forecast.max(),
        'std': forecast.std()
    }
    
    # Calculate overall trend (linear regression)
    x = np.arange(len(forecast))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, forecast.values)
    
    trend_analysis = {
        'slope': slope,
        'growth_rate': slope / intercept if intercept != 0 else None,
        'direction': 'upward' if slope > 0 else 'downward' if slope < 0 else 'flat',
        'r_squared': r_value ** 2,
        'p_value': p_value
    }
    
    # Detect seasonality if enough data points
    if len(forecast) >= 12:
        # Check for autocorrelation at typical seasonal lags
        autocorrelation = {}
        for lag in [3, 4, 6, 12]:
            if len(forecast) > lag:
                # Calculate lag-n autocorrelation
                autocorr = forecast.autocorr(lag=lag)
                autocorrelation[f'lag_{lag}'] = autocorr
    else:
        autocorrelation = {'insufficient_data': True}
    
    return {
        'statistics': stats,
        'trend': trend_analysis,
        'autocorrelation': autocorrelation
    }
