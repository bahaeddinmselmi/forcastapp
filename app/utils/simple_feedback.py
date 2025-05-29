"""
Simple Forecast Feedback Module

A simplified and robust implementation for feedback functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from datetime import datetime
import os

def calculate_accuracy_metrics(forecast: pd.Series, actuals: pd.Series) -> Dict[str, float]:
    """
    Calculate basic accuracy metrics between forecast and actuals
    
    Args:
        forecast: Forecast series with datetime index
        actuals: Actuals series with datetime index
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Align data on dates
    aligned_data = pd.DataFrame({'forecast': forecast, 'actual': actuals})
    aligned_data = aligned_data.dropna()
    
    if len(aligned_data) == 0:
        return {'error': 'No overlapping data points'}
    
    f = aligned_data['forecast'].values
    a = aligned_data['actual'].values
    
    # Calculate basic metrics
    mae = np.mean(np.abs(f - a))
    mape = np.mean(np.abs((a - f) / a)) * 100 if np.all(a != 0) else np.nan
    mse = np.mean((f - a) ** 2)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': round(mae, 2),
        'MAPE': round(mape, 2) if not np.isnan(mape) else 'N/A',
        'RMSE': round(rmse, 2),
        'Samples': len(aligned_data)
    }

def create_comparison_chart(forecasts: Dict[str, Dict[str, Any]], actuals: pd.Series) -> go.Figure:
    """
    Create a chart comparing forecasts with actuals
    
    Args:
        forecasts: Dictionary of forecast results
        actuals: Series of actual values with datetime index
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add actuals
    fig.add_trace(go.Scatter(
        x=actuals.index,
        y=actuals.values,
        mode='lines+markers',
        name='Actual Values',
        line=dict(color='black', width=2)
    ))
    
    # Add each forecast
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'teal']
    for i, (model_name, forecast_dict) in enumerate(forecasts.items()):
        if 'forecast' not in forecast_dict:
            continue
        
        forecast = forecast_dict['forecast']
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color=color)
        ))
    
    # Update layout
    fig.update_layout(
        title='Forecast vs Actual Comparison',
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )
    
    return fig

def calculate_all_metrics(forecasts: Dict[str, Dict[str, Any]], actuals: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy metrics for all forecast models
    
    Args:
        forecasts: Dictionary of forecast results by model
        actuals: Series of actual values
        
    Returns:
        Dictionary of metrics by model
    """
    results = {}
    
    for model_name, forecast_dict in forecasts.items():
        if 'forecast' not in forecast_dict:
            continue
            
        forecast = forecast_dict['forecast']
        metrics = calculate_accuracy_metrics(forecast, actuals)
        results[model_name] = metrics
    
    return results

def save_actuals_data(actuals: pd.Series, file_path: str = "actual_results.csv") -> str:
    """
    Save actuals data to a CSV file
    
    Args:
        actuals: Series of actual values
        file_path: Path to save the file
        
    Returns:
        Path to the saved file
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame({'date': actuals.index, 'actual': actuals.values})
    
    # Save
    df.to_csv(file_path, index=False)
    
    return file_path
