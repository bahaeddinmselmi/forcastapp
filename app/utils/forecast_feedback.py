"""
Forecast Feedback Module

This module provides functionality to:
1. Compare forecasts with actual results
2. Calculate forecast accuracy
3. Retrain models with new data
4. Track model improvement over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import pickle
from datetime import datetime

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from utils.prophet_enhanced import enhanced_prophet_forecast


class ForecastFeedback:
    """Class to handle forecast feedback and model retraining"""
    
    def __init__(self, 
                feedback_dir: str = "forecast_feedback",
                history_size: int = 10):
        """
        Initialize the feedback system
        
        Args:
            feedback_dir: Directory to store feedback and trained models
            history_size: Number of past feedback iterations to track
        """
        self.feedback_dir = feedback_dir
        self.history_size = history_size
        self.current_accuracy = {}
        self.accuracy_history = []
        
        # Create feedback directory if it doesn't exist
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Load existing history if available
        self._load_history()
    
    def _load_history(self):
        """Load accuracy history from storage if available"""
        history_path = os.path.join(self.feedback_dir, "accuracy_history.pkl")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    self.accuracy_history = pickle.load(f)
            except:
                self.accuracy_history = []
    
    def _save_history(self):
        """Save accuracy history to storage"""
        history_path = os.path.join(self.feedback_dir, "accuracy_history.pkl")
        try:
            with open(history_path, 'wb') as f:
                pickle.dump(self.accuracy_history, f)
        except:
            print("Error saving accuracy history")
    
    def process_feedback(self, 
                        forecasts: Dict[str, Dict[str, Any]], 
                        actuals: pd.Series,
                        retrain: bool = True) -> Dict[str, Any]:
        """
        Process forecast feedback by comparing with actuals
        
        Args:
            forecasts: Dictionary of forecast models and their results
            actuals: Series of actual values
            retrain: Whether to retrain models with new data
            
        Returns:
            Dictionary with feedback results and metrics
        """
        results = {}
        metrics = {}
        
        # Make sure actuals index is DatetimeIndex
        if not isinstance(actuals.index, pd.DatetimeIndex):
            try:
                actuals.index = pd.DatetimeIndex(actuals.index)
            except:
                raise ValueError("Actuals index must be convertible to DatetimeIndex")
        
        # Calculate metrics for each model
        for model_name, forecast_dict in forecasts.items():
            if 'forecast' not in forecast_dict:
                continue
                
            forecast = forecast_dict['forecast']
            
            # Align forecast with actuals by date
            aligned_forecast, aligned_actuals = self._align_data(forecast, actuals)
            
            if len(aligned_forecast) == 0 or len(aligned_actuals) == 0:
                metrics[model_name] = {
                    'error': 'No overlapping data points between forecast and actuals'
                }
                continue
            
            # Calculate accuracy metrics
            model_metrics = self._calculate_metrics(aligned_forecast, aligned_actuals)
            metrics[model_name] = model_metrics
        
        # Store current accuracy
        self.current_accuracy = metrics
        
        # Add to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        self.accuracy_history.append(history_entry)
        
        # Keep only the last N entries
        if len(self.accuracy_history) > self.history_size:
            self.accuracy_history = self.accuracy_history[-self.history_size:]
        
        # Save history
        self._save_history()
        
        # Retrain models if requested
        retrained_models = {}
        if retrain:
            retrained_models = self._retrain_models(forecasts, actuals)
            
        # Prepare results
        results = {
            'metrics': metrics,
            'retrained_models': retrained_models
        }
        
        return results
    
    def _align_data(self, 
                  forecast: pd.Series, 
                  actuals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Align forecast and actuals by date
        
        Args:
            forecast: Forecast series
            actuals: Actual values series
            
        Returns:
            Tuple of aligned forecast and actuals
        """
        # Ensure both have DatetimeIndex
        if not isinstance(forecast.index, pd.DatetimeIndex):
            try:
                forecast.index = pd.DatetimeIndex(forecast.index)
            except:
                raise ValueError("Forecast index must be convertible to DatetimeIndex")
        
        # Find common dates
        common_dates = forecast.index.intersection(actuals.index)
        
        # Return aligned series
        return forecast.loc[common_dates], actuals.loc[common_dates]
    
    def _calculate_metrics(self, 
                         forecast: pd.Series, 
                         actuals: pd.Series) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics
        
        Args:
            forecast: Forecast series
            actuals: Actual values series
            
        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {}
        
        # Mean Absolute Error (MAE)
        metrics['MAE'] = np.mean(np.abs(forecast - actuals))
        
        # Mean Absolute Percentage Error (MAPE)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actuals - forecast) / actuals)) * 100
            metrics['MAPE'] = float(np.nan_to_num(mape, nan=0))
        
        # Root Mean Squared Error (RMSE)
        metrics['RMSE'] = np.sqrt(np.mean(np.square(forecast - actuals)))
        
        # Mean Bias Error (MBE)
        metrics['MBE'] = np.mean(forecast - actuals)
        
        # R-squared (coefficient of determination)
        ss_total = np.sum((actuals - np.mean(actuals)) ** 2)
        ss_residual = np.sum((actuals - forecast) ** 2)
        if ss_total > 0:
            metrics['R2'] = 1 - (ss_residual / ss_total)
        else:
            metrics['R2'] = 0
        
        return metrics
    
    def _retrain_models(self, 
                      forecasts: Dict[str, Dict[str, Any]], 
                      actuals: pd.Series) -> Dict[str, Any]:
        """
        Retrain models with new actual data
        
        Args:
            forecasts: Dictionary of forecast models
            actuals: Series of actual values
            
        Returns:
            Dictionary of retrained models
        """
        retrained = {}
        
        # Check for Prophet model
        if 'Prophet' in forecasts:
            retrained['Prophet'] = self._retrain_prophet(forecasts['Prophet'], actuals)
        
        # TODO: Add retraining for other model types
        
        return retrained
    
    def _retrain_prophet(self, 
                       prophet_forecast: Dict[str, Any], 
                       actuals: pd.Series) -> Dict[str, Any]:
        """
        Retrain Prophet model with new data
        
        Args:
            prophet_forecast: Prophet forecast dictionary
            actuals: New actual values
            
        Returns:
            Dictionary with retrained model info
        """
        try:
            # Convert actuals to Prophet format
            new_data = pd.DataFrame({
                'ds': actuals.index,
                'y': actuals.values
            })
            
            # Get parameters from the original model if available
            params = {}
            if 'parameters' in prophet_forecast:
                params = prophet_forecast['parameters']
            
            # Create a new Prophet model (Prophet can only be fit once)
            new_model = Prophet(
                yearly_seasonality=params.get('yearly_seasonality', 'auto'),
                weekly_seasonality=params.get('weekly_seasonality', 'auto'),
                daily_seasonality=params.get('daily_seasonality', 'auto'),
                seasonality_mode=params.get('seasonality_mode', 'additive'),
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05)
            )
            
            # Fit the new model with the updated data
            new_model.fit(new_data)
            
            # Save updated model
            model_path = os.path.join(self.feedback_dir, "prophet_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(new_model, f)
                
            return {
                'status': 'success',
                'model_path': model_path,
                'model': new_model  # Include the model in the return dictionary
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error retraining Prophet model: {str(e)}'
            }
    
    def get_accuracy_trend(self) -> Dict[str, List[float]]:
        """
        Get the trend of forecast accuracy over time
        
        Returns:
            Dictionary of accuracy metrics over time
        """
        if not self.accuracy_history:
            return {}
        
        # Prepare trend data
        trend = {}
        
        # Extract timestamps
        timestamps = [entry['timestamp'] for entry in self.accuracy_history]
        trend['timestamps'] = timestamps
        
        # For each model, extract metrics over time
        models = set()
        for entry in self.accuracy_history:
            for model in entry['metrics'].keys():
                models.add(model)
        
        for model in models:
            trend[model] = {}
            
            # For each metric, extract values over time
            metrics = ['MAE', 'MAPE', 'RMSE', 'R2']
            for metric in metrics:
                values = []
                for entry in self.accuracy_history:
                    if model in entry['metrics'] and metric in entry['metrics'][model]:
                        values.append(entry['metrics'][model][metric])
                    else:
                        values.append(None)
                
                trend[model][metric] = values
        
        return trend
    
    def create_accuracy_chart(self) -> go.Figure:
        """
        Create a chart showing accuracy trends over time
        
        Returns:
            Plotly figure with accuracy trends
        """
        trend_data = self.get_accuracy_trend()
        
        if not trend_data:
            # Create empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="No accuracy history available yet",
                xaxis_title="Time",
                yaxis_title="Accuracy"
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Add MAPE trends for each model
        for model in trend_data.keys():
            if model == 'timestamps':
                continue
                
            if 'MAPE' in trend_data[model]:
                fig.add_trace(go.Scatter(
                    x=trend_data['timestamps'],
                    y=trend_data[model]['MAPE'],
                    mode='lines+markers',
                    name=f'{model} MAPE'
                ))
        
        # Update layout
        fig.update_layout(
            title="Forecast Accuracy Trend (MAPE)",
            xaxis_title="Time",
            yaxis_title="MAPE (%)",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified'
        )
        
        return fig
        
    def create_comparison_chart(self, 
                              forecasts: Dict[str, Dict[str, Any]], 
                              actuals: pd.Series) -> go.Figure:
        """
        Create a chart comparing forecasts with actuals
        
        Args:
            forecasts: Dictionary of forecast models
            actuals: Series of actual values
            
        Returns:
            Plotly figure with comparison
        """
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=actuals.index,
            y=actuals,
            mode='lines+markers',
            name='Actual Values',
            line=dict(color='black', width=2)
        ))
        
        # Add forecasts
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, (model_name, forecast_dict) in enumerate(forecasts.items()):
            if 'forecast' not in forecast_dict:
                continue
                
            forecast = forecast_dict['forecast']
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=color)
            ))
            
            # Add confidence intervals if available
            if 'lower_bound' in forecast_dict and 'upper_bound' in forecast_dict:
                lower = forecast_dict['lower_bound']
                upper = forecast_dict['upper_bound']
                
                # Add confidence interval as a shaded area
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast.index, forecast.index[::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({",".join(map(str, [int(int(color[1:3], 16)), int(color[3:5], 16), int(color[5:7], 16), 0.2]))})' if color.startswith('#') else f'rgba(100,100,100,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{model_name} Confidence Interval',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title="Forecast vs Actual Comparison",
            xaxis_title="Date",
            yaxis_title="Value",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified'
        )
        
        return fig


def add_actuals_and_retrain(forecasts: Dict[str, Dict[str, Any]], 
                           actuals: pd.Series,
                           feedback_dir: str = "forecast_feedback",
                           history_size: int = 10) -> Dict[str, Any]:
    """
    Convenience function to add actuals and retrain models
    
    Args:
        forecasts: Dictionary of forecast models
        actuals: Series of actual values
        feedback_dir: Directory to store feedback
        history_size: Number of history entries to keep
        
    Returns:
        Dictionary with feedback results
    """
    feedback = ForecastFeedback(feedback_dir=feedback_dir, history_size=history_size)
    return feedback.process_feedback(forecasts, actuals, retrain=True)


def create_forecast_comparison_with_actuals(forecasts: Dict[str, Dict[str, Any]], 
                                         actuals: pd.Series) -> go.Figure:
    """
    Create a comparison chart with forecasts and actuals
    
    Args:
        forecasts: Dictionary of forecast models
        actuals: Series of actual values
        
    Returns:
        Plotly figure with comparison
    """
    feedback = ForecastFeedback()
    return feedback.create_comparison_chart(forecasts, actuals)
    

def calculate_forecast_accuracy(forecasts: Dict[str, Dict[str, Any]], 
                              actuals: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate forecast accuracy metrics
    
    Args:
        forecasts: Dictionary of forecast models
        actuals: Series of actual values
        
    Returns:
        Dictionary of accuracy metrics by model
    """
    feedback = ForecastFeedback()
    results = feedback.process_feedback(forecasts, actuals, retrain=False)
    return results['metrics']
