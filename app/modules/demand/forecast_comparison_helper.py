"""
Helper functions for forecast comparisons with improved Series/DataFrame handling
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Union

def safe_create_forecast_comparison(
    forecasts: Dict[str, Dict[str, Any]],
    historical_data: Union[pd.DataFrame, pd.Series],
    target_col: Optional[str] = None,
    title: str = "Forecast Comparison"
) -> go.Figure:
    """
    Create a comparison plot of multiple forecast models with safe Series/DataFrame handling.
    
    Args:
        forecasts: Dictionary of forecast model results
        historical_data: Historical data as DataFrame or Series
        target_col: Name of the target column (used if historical_data is DataFrame)
        title: Plot title
        
    Returns:
        Plotly figure with forecast comparison
    """
    # Create figure
    fig = go.Figure()
    
    # Handle historical data, safely extracting the target series
    if historical_data is not None and not historical_data.empty:
        if isinstance(historical_data, pd.DataFrame):
            # For DataFrame, check if target_col exists
            if target_col is not None and target_col in historical_data.columns:
                historical_series = historical_data[target_col]
            elif historical_data.shape[1] > 0:
                # Use first column if target_col not found
                historical_series = historical_data.iloc[:, 0]
                print(f"Target column '{target_col}' not found. Using first column.")
            else:
                # Empty DataFrame
                historical_series = None
        else:
            # If already a Series, use directly
            historical_series = historical_data
            
        # Add historical line if we have data
        if historical_series is not None:
            fig.add_trace(go.Scatter(
                x=historical_series.index,
                y=historical_series.values,
                mode='lines',
                name='Historical Data',
                line=dict(color='gray', width=2, dash='dash')
            ))
    
    # Plot each forecast model
    colors = px.colors.qualitative.Plotly
    for i, (model_name, result) in enumerate(forecasts.items()):
        # Skip if result is None
        if result is None:
            continue
            
        # Skip if forecast is missing
        if 'forecast' not in result:
            continue
            
        forecast = result['forecast']
        if forecast is None or len(forecast) == 0:
            continue
            
        # Use color cycling
        color_idx = i % len(colors)
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name=model_name,
            line=dict(color=colors[color_idx], width=2)
        ))
        
        # Add confidence intervals if available
        if 'lower_bound' in result and 'upper_bound' in result:
            lower_bound = result['lower_bound']
            upper_bound = result['upper_bound']
            
            if lower_bound is not None and upper_bound is not None:
                # Create a filled area for confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                    y=upper_bound.values.tolist() + lower_bound.values.tolist()[::-1],
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[color_idx])) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} Confidence Interval',
                    showlegend=False
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )
    
    return fig
