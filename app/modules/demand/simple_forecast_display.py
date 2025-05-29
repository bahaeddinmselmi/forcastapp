"""
Simple Forecast Value Display
This module provides a streamlined function to display forecast values in the IBP system
"""

import streamlit as st
import pandas as pd

def enhance_figure_tooltips(fig):
    """
    Enhance a Plotly figure's tooltips to show exact values
    
    Args:
        fig: The Plotly figure to enhance
        
    Returns:
        The enhanced figure
    """
    # Add detailed tooltip styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Configure hover template to show exact values
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'hovertemplate') and trace.hovertemplate is not None:
            fig.data[i].hovertemplate = '%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>'
    
    return fig

def show_forecast_values(forecasts=None, title="Forecast Values"):
    """
    Display the exact forecast values in a simple table with tabs for each model
    """
    if forecasts is None or len(forecasts) == 0:
        if 'forecasts' in st.session_state:
            forecasts = st.session_state['forecasts']
        else:
            st.info("No forecast data available yet. Please run forecast models first.")
            return
    
    st.subheader(title)
    
    # Create tabs for each model
    model_names = list(forecasts.keys())
    if not model_names:
        st.info("No forecast models available yet.")
        return
        
    tabs = st.tabs(model_names)
    
    # Display each model's values in its tab
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            model_data = forecasts[model_name]
            if 'forecast' in model_data and isinstance(model_data['forecast'], pd.Series):
                # Create a dataframe with the forecast data
                df = pd.DataFrame()
                df['Date'] = model_data['forecast'].index
                df['Forecast Value'] = model_data['forecast'].values.round(2)
                
                # Add confidence intervals if available
                if 'lower_bound' in model_data and 'upper_bound' in model_data:
                    if isinstance(model_data['lower_bound'], pd.Series):
                        df['Lower Bound'] = model_data['lower_bound'].values.round(2)
                    if isinstance(model_data['upper_bound'], pd.Series):
                        df['Upper Bound'] = model_data['upper_bound'].values.round(2)
                
                # Display the dataframe
                st.dataframe(df, use_container_width=True)
                
                # Add download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {model_name} Values as CSV",
                    data=csv,
                    file_name=f"{model_name.lower().replace(' ', '_')}_forecast.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No forecast data available for {model_name}")

def show_cumulative_values(cum_forecasts=None, title="Cumulative Forecast Values"):
    """
    Display the exact cumulative forecast values in a simple table with tabs for each model
    """
    if cum_forecasts is None or len(cum_forecasts) == 0:
        if 'cumulative_forecasts' in st.session_state:
            cum_forecasts = st.session_state['cumulative_forecasts']
        else:
            st.info("No cumulative forecast data available yet. Please run forecast models first.")
            return
    
    st.subheader(title)
    
    # Create tabs for each model
    model_names = list(cum_forecasts.keys())
    if not model_names:
        st.info("No cumulative forecast models available yet.")
        return
        
    tabs = st.tabs(model_names)
    
    # Display each model's values in its tab
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            model_data = cum_forecasts[model_name]
            if 'forecast' in model_data and isinstance(model_data['forecast'], pd.Series):
                # Create a dataframe with the forecast data
                df = pd.DataFrame()
                df['Date'] = model_data['forecast'].index
                df['Cumulative Value'] = model_data['forecast'].values.round(2)
                
                # Add confidence intervals if available
                if 'lower_bound' in model_data and 'upper_bound' in model_data:
                    if isinstance(model_data['lower_bound'], pd.Series):
                        df['Lower Bound'] = model_data['lower_bound'].values.round(2)
                    if isinstance(model_data['upper_bound'], pd.Series):
                        df['Upper Bound'] = model_data['upper_bound'].values.round(2)
                
                # Display the dataframe
                st.dataframe(df, use_container_width=True)
                
                # Add download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {model_name} Cumulative Values as CSV",
                    data=csv,
                    file_name=f"{model_name.lower().replace(' ', '_')}_cumulative.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No cumulative data available for {model_name}")
