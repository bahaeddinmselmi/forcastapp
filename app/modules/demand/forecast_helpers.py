"""
Helper functions for forecast visualization and value display for the IBP forecasting system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Any, List, Optional
from utils.export import create_excel_download_link, create_full_report_download_link

def create_value_display(forecasts, is_cumulative=False):
    """
    Create a display of forecast values in tabular format
    
    Args:
        forecasts: Dictionary of forecast model results
        is_cumulative: Whether these are cumulative forecasts
    """
    type_label = "Cumulative " if is_cumulative else ""
    
    if not forecasts:
        st.info(f"No {type_label.lower()}forecast data available")
        return
        
    # Create tabs for each model
    model_names = list(forecasts.keys())
    tabs = st.tabs(model_names)
    
    # For each model, create a table of values
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            result = forecasts[model_name]
            if 'forecast' in result and isinstance(result['forecast'], pd.Series):
                # Format data as a dataframe
                df = pd.DataFrame()
                
                # Format dates nicely
                if isinstance(result['forecast'].index[0], (datetime, pd.Timestamp)):
                    df['Date'] = [d.strftime('%Y-%m-%d') for d in result['forecast'].index]
                else:
                    df['Date'] = result['forecast'].index
                    
                # Add the forecast values, rounded to 2 decimal places
                df[f'{type_label}Forecast Value'] = result['forecast'].values.round(2)
                
                # Add confidence intervals if available
                if all(k in result for k in ['lower_bound', 'upper_bound']):
                    if isinstance(result['lower_bound'], pd.Series):
                        df['Lower Bound'] = result['lower_bound'].values.round(2)
                    if isinstance(result['upper_bound'], pd.Series):
                        df['Upper Bound'] = result['upper_bound'].values.round(2)
                
                # Display table and download option
                st.dataframe(df, use_container_width=True)
                
                # Add a download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {model_name} {type_label}Values",
                    data=csv,
                    file_name=f"{model_name.lower().replace(' ', '_')}_{'cumulative_' if is_cumulative else ''}forecast.csv",
                    mime="text/csv",
                    key=f"download_{model_name}_{is_cumulative}"
                )
            else:
                st.info(f"No forecast data available for {model_name}")

def display_forecast_summary(forecasts: Dict[str, Dict[str, Any]], evaluation_metrics: Optional[pd.DataFrame] = None, best_model: Optional[str] = None):
    """
    Display a summary of forecast information with key statistics
    
    Args:
        forecasts: Dictionary of forecast model results
        evaluation_metrics: Optional evaluation metrics dataframe
        best_model: Optional name of the best model
    """
    if not forecasts:
        st.info("No forecast data available yet")
        return
    
    # Get best model if not provided but available in session state
    if best_model is None and 'best_model' in st.session_state:
        best_model = st.session_state['best_model']
    
    # Create summary metrics for each model
    summary_data = []
    
    for model_name, result in forecasts.items():
        if 'forecast' in result and isinstance(result['forecast'], pd.Series):
            forecast = result['forecast']
            
            # Calculate basic statistics
            model_stats = {
                'Model': model_name,
                'Min': forecast.min().round(2),
                'Max': forecast.max().round(2),
                'Mean': forecast.mean().round(2),
                'Total': forecast.sum().round(2),
                'Points': len(forecast)
            }
            
            # Add evaluation metrics if available
            if evaluation_metrics is not None and not evaluation_metrics.empty:
                model_metrics = evaluation_metrics[evaluation_metrics['Model'] == model_name]
                if not model_metrics.empty:
                    if 'RMSE' in model_metrics.columns:
                        model_stats['RMSE'] = model_metrics['RMSE'].values[0].round(2)
                    if 'MAPE' in model_metrics.columns:
                        model_stats['MAPE'] = model_metrics['MAPE'].values[0].round(2)
            
            summary_data.append(model_stats)
    
    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Highlight the best model if available
        if best_model and best_model in summary_df['Model'].values:
            # Add visual indicator for best model
            summary_df['Best Model'] = summary_df['Model'].apply(lambda x: '✓ Recommended' if x == best_model else '')
            
            # Apply styling to highlight best model
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.dataframe(summary_df, use_container_width=True)
        
        # Add Excel export options
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(create_excel_download_link(summary_df, "Forecast Summary", "Forecast Summary"), unsafe_allow_html=True)
        with col2:
            # Create metadata for full report
            metadata = {
                "Generated on": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Number of Models": len(summary_data),
                "Best Model": best_model if best_model else "Not determined",
                "Forecast Period": f"{summary_data[0]['Points'] if summary_data else 0} periods"
            }
            
            # Generate full report download link
            st.markdown(create_full_report_download_link(
                summary_df, 
                "Forecast Analysis Report", 
                None,  # We'll add chart support later
                metadata
            ), unsafe_allow_html=True)
        
        # Display key insights about the forecasts
        with st.expander("📊 Forecast Insights"):
            # Find the model with highest and lowest forecast
            highest_forecast = max(summary_data, key=lambda x: x['Max'])
            lowest_forecast = min(summary_data, key=lambda x: x['Min'])
            
            # Display insights
            cols = st.columns(2)
            with cols[0]:
                st.metric("Highest Forecast Value", f"{highest_forecast['Max']}", f"Model: {highest_forecast['Model']}")
                if 'MAPE' in highest_forecast:
                    st.caption(f"MAPE: {highest_forecast['MAPE']}%")
            
            with cols[1]:
                if best_model:
                    best_stats = next((x for x in summary_data if x['Model'] == best_model), None)
                    if best_stats:
                        st.metric("Recommended Model Total", f"{best_stats['Total']}", f"Mean: {best_stats['Mean']}")
                        if 'MAPE' in best_stats:
                            st.caption(f"MAPE: {best_stats['MAPE']}%")
                else:
                    st.metric("Lowest Forecast Value", f"{lowest_forecast['Min']}", f"Model: {lowest_forecast['Model']}")
                    if 'MAPE' in lowest_forecast:
                        st.caption(f"MAPE: {lowest_forecast['MAPE']}%")
    else:
        st.info("No forecast data available for summary")


def enhance_plot_tooltips(fig):
    """
    Enhance tooltips on a plotly figure to show exact values
    
    Args:
        fig: Plotly figure to enhance
    
    Returns:
        Enhanced figure or None if input was None
    """
    # Check if figure is None
    if fig is None:
        return None
        
    # Enhance hover info display
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Update hover template to show exact values
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'hovertemplate') and trace.hovertemplate is not None:
            fig.data[i].hovertemplate = '%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>'
    
    return fig


def create_forecast_comparison(forecasts: Dict[str, Dict[str, Any]], historical_data: pd.DataFrame = None, target_col: str = None):
    """
    Create a comparative visualization of different forecast models side by side
    
    Args:
        forecasts: Dictionary of forecast model results
        historical_data: Optional historical data for context
        target_col: Target column name in historical data
        
    Returns:
        Plotly figure object with the comparison visualization
    """
    if not forecasts:
        return None
        
    # Create a new figure
    fig = go.Figure()
    
    # First, add historical data if available
    if historical_data is not None and target_col is not None:
        if not historical_data.empty:
            # Get the historical series
            historical_series = historical_data[target_col]
            
            # Add historical line
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
            
        # Skip if forecast isn't available in the result
        if not isinstance(result, dict) or 'forecast' not in result:
            continue
            
        # Skip if forecast isn't a valid pandas Series
        if not isinstance(result['forecast'], pd.Series):
            continue
            
        # Get forecast data safely
        try:
            forecast_x = result['forecast'].index
            forecast_y = result['forecast'].values
            color_idx = i % len(colors)  # Cycle through colors if more models than colors
            
            # Plot forecast line
            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast_y,
                mode='lines', name=model_name,
                line=dict(color=colors[color_idx], width=3)
            ))
        except Exception as e:
            print(f"Error plotting {model_name}: {str(e)}")
            continue
    
    # Update layout with better styling
    fig.update_layout(
        title='Forecast Model Comparison',
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Enhance tooltips
    fig = enhance_plot_tooltips(fig)
    
    return fig


def create_side_by_side_comparison(standard_forecasts: Dict[str, Dict[str, Any]], cumulative_forecasts: Dict[str, Dict[str, Any]], best_model: Optional[str] = None):
    """
    Create a side-by-side comparison of regular vs cumulative forecasts
    
    Args:
        standard_forecasts: Dictionary of regular forecast model results
        cumulative_forecasts: Dictionary of cumulative forecast model results
        best_model: Optional name of the best model to highlight
        
    Returns:
        Plotly figure object with the comparison visualization
    """
    if not standard_forecasts or not cumulative_forecasts:
        return None
        
    # Create the subplots - 1 row, 2 columns
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=("Regular Forecast", "Cumulative Forecast"),
        shared_yaxes=False,
        horizontal_spacing=0.05
    )
    
    # Add forecast models - focus on the best model if available
    colors = px.colors.qualitative.Plotly
    models_to_display = list(standard_forecasts.keys())
    
    # If best model is available, make it the first one to ensure it gets a distinctive color
    if best_model and best_model in models_to_display:
        models_to_display.remove(best_model)
        models_to_display = [best_model] + models_to_display
    
    for i, model_name in enumerate(models_to_display):
        color_idx = i % len(colors)  # Cycle through colors if more models than colors
        line_width = 3 if model_name == best_model else 2
        
        # Add to regular forecast subplot
        if model_name in standard_forecasts:
            result = standard_forecasts[model_name]
            if 'forecast' in result and isinstance(result['forecast'], pd.Series):
                fig.add_trace(
                    go.Scatter(
                        x=result['forecast'].index,
                        y=result['forecast'].values,
                        mode='lines',
                        name=model_name,
                        line=dict(color=colors[color_idx], width=line_width)
                    ),
                    row=1, col=1
                )
        
        # Add to cumulative forecast subplot
        if model_name in cumulative_forecasts:
            result = cumulative_forecasts[model_name]
            if 'forecast' in result and isinstance(result['forecast'], pd.Series):
                fig.add_trace(
                    go.Scatter(
                        x=result['forecast'].index,
                        y=result['forecast'].values,
                        mode='lines',
                        name=f"{model_name} (Cumulative)",
                        line=dict(color=colors[color_idx], width=line_width),
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    # Update layout with better styling
    fig.update_layout(
        title='Forecast Comparison: Regular vs. Cumulative',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    # Update x and y axis titles
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Value", row=1, col=2)
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Enhance tooltips
    fig = enhance_plot_tooltips(fig)
    
    return fig
