"""
Forecast Feedback UI Module

This module provides the UI components for the forecast feedback feature,
allowing users to upload actual results and improve model accuracy over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

from utils.forecast_feedback import (
    ForecastFeedback,
    add_actuals_and_retrain,
    create_forecast_comparison_with_actuals,
    calculate_forecast_accuracy
)

def show_forecast_feedback_section():
    """
    Display the forecast feedback section in the Streamlit UI.
    This allows users to:
    1. Upload actual results
    2. Compare forecasts vs actuals
    3. Retrain models with new data
    4. Track accuracy improvements
    """
    st.markdown("### Forecast Feedback & Model Learning")
    
    st.markdown("""
    Improve your forecast models by uploading actual results. 
    The system will compare forecasts against actuals, calculate accuracy metrics,
    and automatically retrain models to improve future forecasts.
    """)
    
    # Check if forecasts are available
    if 'forecasts' not in st.session_state:
        st.warning("⚠️ Please generate forecasts first in the Forecast Models tab.")
        return
    
    # Get current forecasts
    forecasts = st.session_state['forecasts']
    
    # Initialize feedback system
    feedback_dir = os.path.join("data", "feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_system = ForecastFeedback(feedback_dir=feedback_dir)
    
    # Show tabs for different feedback functions
    feedback_tab1, feedback_tab2, feedback_tab3 = st.tabs([
        "📊 Upload Actuals", 
        "📈 Forecast vs Actuals", 
        "📉 Accuracy Trends"
    ])
    
    with feedback_tab1:
        st.markdown("#### Upload Actual Results")
        st.markdown("""
        Upload a file containing the actual values to compare with your forecasts.
        The file should have dates in the first column and actual values in the second column.
        """)
        
        # File uploader for actuals
        uploaded_actuals = st.file_uploader(
            "Upload a CSV or Excel file with actual values",
            type=['csv', 'xlsx', 'xls'],
            key="actuals_uploader"
        )
        
        # Manual input option
        st.markdown("#### Or Enter Actual Values Manually")
        
        # Get date range from forecasts to show in the UI
        forecast_dates = []
        if forecasts:
            # Find the first available forecast
            for model_name, forecast_dict in forecasts.items():
                if 'forecast' in forecast_dict and forecast_dict['forecast'] is not None:
                    forecast_dates = forecast_dict['forecast'].index
                    break
        
        # Create a sample dataframe with forecast dates
        sample_df = pd.DataFrame({
            "Date": forecast_dates,
            "Actual_Value": [None] * len(forecast_dates)
        })
        
        # Show editable dataframe - compatible with older Streamlit versions
        if not forecast_dates.empty:
            # Check if data_editor is available (Streamlit >= 1.10.0)
            if hasattr(st, 'data_editor'):
                edited_df = st.data_editor(
                    sample_df,
                    key="actuals_editor"
                )
            else:
                # Fallback for older Streamlit versions
                st.write("Enter Actual Values:")
                st.dataframe(sample_df)
                
                # Simplified direct approach for adding values
                st.write("Enter actual values for specific dates:")
                
                # Initialize session state for storing actuals if not already present
                if 'actual_values_dict' not in st.session_state:
                    st.session_state.actual_values_dict = {}
                
                # Create two columns for date and value inputs
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Create date selector
                    date_options = [d.strftime('%Y-%m-%d') for d in forecast_dates]
                    selected_date = st.selectbox("Select Date:", date_options)
                
                with col2:
                    # Value input
                    actual_value = st.number_input("Actual Value:", min_value=0.0, step=0.01)
                
                with col3:
                    # Add button outside of form for immediate action
                    if st.button("Add Value", key="add_actual_value"):
                        # Store the value in session state dictionary
                        st.session_state.actual_values_dict[selected_date] = actual_value
                        
                        # Show success message
                        st.success(f"Added actual value {actual_value} for {selected_date}")
                
                # Display current actual values
                if st.session_state.actual_values_dict:
                    st.write("Current Entered Values:")
                    
                    # Create a DataFrame from the dictionary
                    actual_entries = []
                    for date_str, value in st.session_state.actual_values_dict.items():
                        actual_entries.append({"Date": date_str, "Actual_Value": value})
                    
                    # Convert to DataFrame for display
                    if actual_entries:
                        edited_df = pd.DataFrame(actual_entries)
                        st.dataframe(edited_df)
                
                # Use the session state dataframe
                if 'edited_actuals' in st.session_state:
                    edited_df = st.session_state['edited_actuals']
                    st.write("Current Actual Values:")
                    st.dataframe(edited_df)
                else:
                    edited_df = sample_df
        else:
            st.info("No forecast dates available. Please generate forecasts first.")
            edited_df = pd.DataFrame()
        
        # Process actuals data
        actuals_df = None
        
        if uploaded_actuals is not None:
            try:
                # Determine file type and read
                if uploaded_actuals.name.endswith('.csv'):
                    actuals_df = pd.read_csv(uploaded_actuals)
                else:
                    actuals_df = pd.read_excel(uploaded_actuals)
                
                # Ensure first column is date
                date_col = actuals_df.columns[0]
                value_col = actuals_df.columns[1]
                
                # Convert to datetime
                actuals_df[date_col] = pd.to_datetime(actuals_df[date_col])
                
                # Set index to date
                actuals_df = actuals_df.set_index(date_col)
                
                # Create series from dataframe
                actuals_series = actuals_df[value_col]
                
                # Store actuals in session state
                st.session_state['actuals'] = actuals_series
                
                st.success(f"Successfully loaded {len(actuals_series)} actual values from file.")
            except Exception as e:
                st.error(f"Error loading actuals file: {str(e)}")
        
        # Process manually entered data if available (using the new approach)
        if 'actual_values_dict' in st.session_state and st.session_state.actual_values_dict:
            # Convert the dictionary to a Series
            dates = [pd.to_datetime(date_str) for date_str in st.session_state.actual_values_dict.keys()]
            values = list(st.session_state.actual_values_dict.values())
            
            # Create a Series with datetime index
            actuals_series = pd.Series(values, index=dates)
            
            # Store actuals in session state
            st.session_state['actuals'] = actuals_series
            
            # Show this message only if we didn't already show a success message for file upload
            if uploaded_actuals is None:
                st.success(f"Successfully recorded {len(actuals_series)} actual values.")
        elif uploaded_actuals is None and ('actual_values_dict' not in st.session_state or not st.session_state.actual_values_dict):
            st.info("Please enter at least one actual value.")
        
        # Submit button to process actuals
        if st.button("Process Actuals & Retrain Models"):
            if 'actuals' in st.session_state and not st.session_state['actuals'].empty:
                with st.spinner("Processing actuals and retraining models..."):
                    try:
                        # Get actuals
                        actuals = st.session_state['actuals']
                        
                        # Process feedback
                        results = add_actuals_and_retrain(
                            forecasts=forecasts,
                            actuals=actuals,
                            feedback_dir=feedback_dir
                        )
                        
                        # Store metrics
                        st.session_state['accuracy_metrics'] = results['metrics']
                        
                        # Show results summary
                        st.success("✅ Successfully processed actuals and retrained models!")
                        
                        # Display metrics
                        for model_name, metrics in results['metrics'].items():
                            if isinstance(metrics, dict) and 'error' not in metrics:
                                st.markdown(f"**{model_name} Accuracy Metrics:**")
                                metrics_df = pd.DataFrame({
                                    "Metric": metrics.keys(),
                                    "Value": metrics.values()
                                })
                                st.dataframe(metrics_df.reset_index(drop=True))
                    
                    except Exception as e:
                        st.error(f"Error processing actuals: {str(e)}")
            else:
                st.warning("No actual values available. Please upload or enter actual values first.")
    
    with feedback_tab2:
        st.markdown("#### Forecast vs Actuals Comparison")
        
        if 'actuals' in st.session_state and not st.session_state['actuals'].empty:
            # Create comparison chart
            with st.spinner("Creating comparison chart..."):
                try:
                    actuals = st.session_state['actuals']
                    comparison_fig = create_forecast_comparison_with_actuals(
                        forecasts=forecasts,
                        actuals=actuals
                    )
                    
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Show accuracy metrics if available
                    if 'accuracy_metrics' in st.session_state:
                        st.markdown("#### Accuracy Metrics")
                        
                        metrics = st.session_state['accuracy_metrics']
                        
                        # Create metrics table
                        metrics_data = []
                        
                        for model_name, model_metrics in metrics.items():
                            if isinstance(model_metrics, dict) and 'error' not in model_metrics:
                                row = {'Model': model_name}
                                row.update(model_metrics)
                                metrics_data.append(row)
                        
                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data)
                            try:
                                st.dataframe(metrics_df.reset_index(drop=True))
                            except:
                                # Fallback for older Streamlit versions
                                st.dataframe(metrics_df.reset_index(drop=True))
                        else:
                            st.info("No accuracy metrics available yet.")
                    
                except Exception as e:
                    st.error(f"Error creating comparison: {str(e)}")
        else:
            st.info("Please upload or enter actual values in the 'Upload Actuals' tab first.")
    
    with feedback_tab3:
        st.markdown("#### Forecast Accuracy Trends")
        st.markdown("""
        Track how your forecast accuracy improves over time as the models learn from your feedback.
        """)
        
        # Get and display accuracy trend
        try:
            trend_fig = feedback_system.create_accuracy_chart()
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Get trend data
            trend_data = feedback_system.get_accuracy_trend()
            
            if trend_data and 'timestamps' in trend_data and len(trend_data['timestamps']) > 0:
                st.markdown("#### Detailed Accuracy History")
                
                # Create history table
                history_data = []
                
                for i, timestamp in enumerate(trend_data['timestamps']):
                    row = {'Timestamp': timestamp}
                    
                    for model in trend_data.keys():
                        if model == 'timestamps':
                            continue
                            
                        if 'MAPE' in trend_data[model]:
                            row[f"{model} MAPE"] = trend_data[model]['MAPE'][i]
                        
                        if 'RMSE' in trend_data[model]:
                            row[f"{model} RMSE"] = trend_data[model]['RMSE'][i]
                    
                    history_data.append(row)
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    try:
                        st.dataframe(history_df.reset_index(drop=True))
                    except:
                        # Fallback for older Streamlit versions
                        st.dataframe(history_df.reset_index(drop=True))
            else:
                st.info("No accuracy history available yet. Provide feedback multiple times to see trends.")
                
        except Exception as e:
            st.error(f"Error displaying accuracy trends: {str(e)}")
            
    # Wrap up with tips
    st.markdown("---")
    st.markdown("""
    #### 💡 Tips for Best Results
    
    - Regularly upload actual values as they become available
    - Compare accuracy across different models to select the best performer
    - Watch for accuracy improvements over time as models learn from your data
    - For best results, provide actuals for at least 3-4 forecast periods
    """)
