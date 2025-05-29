"""
Simple Forecast Feedback UI

A simplified and reliable UI for forecast feedback functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, List, Optional

from utils.simple_feedback import (
    calculate_accuracy_metrics,
    create_comparison_chart,
    calculate_all_metrics,
    save_actuals_data
)

def show_simple_feedback_section():
    """
    Display a simplified and reliable forecast feedback section
    """
    st.markdown("### Forecast Feedback")
    
    # Check if forecasts exist
    if 'forecasts' not in st.session_state:
        st.warning("⚠️ Please generate forecasts first in the Forecast Models tab.")
        return
    
    forecasts = st.session_state['forecasts']
    
    # Show tabs for different functionality
    tab1, tab2, tab3 = st.tabs([
        "📊 Enter Actuals", 
        "📈 Compare Forecasts vs Actuals", 
        "📉 Accuracy Metrics"
    ])
    
    # Initialize session state for actuals if needed
    if 'actual_values' not in st.session_state:
        st.session_state.actual_values = {}
    
    with tab1:
        st.markdown("#### Enter Actual Values")
        
        # File upload option
        st.markdown("##### Upload Actual Results")
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file with actual values (Date in first column, Values in second column)",
            type=['csv', 'xlsx', 'xls'],
            key="actuals_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Determine file type and read
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Get column names
                date_col = df.columns[0]
                value_col = df.columns[1]
                
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Create actuals series
                actuals_series = pd.Series(df[value_col].values, index=df[date_col])
                
                # Store in session state
                st.session_state.actuals_series = actuals_series
                
                # Show success message and preview
                st.success(f"✅ Successfully loaded {len(actuals_series)} actual values.")
                st.write("Preview of loaded data:")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Or manual entry
        st.markdown("##### Or Enter Values Manually")
        
        # Get forecast dates
        forecast_dates = []
        for model_name, forecast_dict in forecasts.items():
            if 'forecast' in forecast_dict and forecast_dict['forecast'] is not None:
                forecast_dates = forecast_dict['forecast'].index
                break
        
        if len(forecast_dates) > 0:
            # Display dates and value inputs in a clean layout
            date_options = [d.strftime('%Y-%m-%d') for d in forecast_dates]
            
            # Create three columns for date, value, and add button
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_date = st.selectbox(
                    "Select Date:", 
                    date_options,
                    key="date_selector"
                )
            
            with col2:
                value = st.number_input(
                    "Actual Value:", 
                    min_value=0.0, 
                    step=0.1,
                    key="value_input"
                )
            
            with col3:
                st.write("")  # Add spacing
                st.write("")  # Add spacing
                add_button = st.button("Add Value", key="add_value_button")
            
            # Handle button click
            if add_button:
                # Store the value
                st.session_state.actual_values[selected_date] = value
                st.success(f"Added value {value} for {selected_date}")
            
            # Display current values
            if st.session_state.actual_values:
                st.markdown("#### Current Values")
                
                # Create DataFrame for display
                data = []
                for date_str, val in st.session_state.actual_values.items():
                    data.append({"Date": date_str, "Value": val})
                
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Convert to Series for processing
                    dates = [pd.to_datetime(d) for d in df["Date"]]
                    values = df["Value"].values
                    actuals_series = pd.Series(values, index=dates)
                    
                    # Store in session state
                    st.session_state.actuals_series = actuals_series
                    
            # Process button - moved outside the condition to always be visible
            st.markdown("---")
            st.markdown("### Process Actual Values")
            
            # Make the button very prominent
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📊 PROCESS ACTUAL VALUES", key="process_actuals_button", use_container_width=True, type="primary"):
                    if 'actuals_series' in st.session_state and len(st.session_state.actuals_series) > 0:
                        st.session_state.processed_actuals = True
                        
                        # Calculate metrics
                        metrics = calculate_all_metrics(forecasts, st.session_state.actuals_series)
                        st.session_state.accuracy_metrics = metrics
                        
                        st.success("✅ Actual values processed successfully!")
                        st.info("Go to the 'Compare Forecasts vs Actuals' and 'Accuracy Metrics' tabs to see results.")
                    else:
                        st.error("⚠️ No actual values to process. Please add some values first.")
            
            st.markdown("""Click this button after you've entered all your actual values to compare them with forecasts 
            and calculate accuracy metrics. Then check the other tabs to view results.""")
            
            # Clear button
            if st.button("Clear All Values", key="clear_values_button"):
                st.session_state.actual_values = {}
                if 'actuals_series' in st.session_state:
                    del st.session_state.actuals_series
                if 'processed_actuals' in st.session_state:
                    del st.session_state.processed_actuals
                st.experimental_rerun()
                    
        else:
            st.info("Please generate forecasts first to see available dates.")
            
    with tab2:
        st.markdown("#### Forecast vs Actuals Comparison")
        
        if 'actuals_series' in st.session_state and 'forecasts' in st.session_state:
            # Create comparison chart
            try:
                fig = create_comparison_chart(forecasts, st.session_state.actuals_series)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download options
                if st.button("Export Comparison Data", key="export_comparison"):
                    # Create a DataFrame with forecasts and actuals
                    export_data = pd.DataFrame(index=st.session_state.actuals_series.index)
                    export_data['Actual'] = st.session_state.actuals_series
                    
                    for model_name, forecast_dict in forecasts.items():
                        if 'forecast' in forecast_dict:
                            # Align to same dates as actuals
                            forecast = forecast_dict['forecast']
                            aligned_forecast = forecast.reindex(export_data.index, method='nearest')
                            export_data[f'{model_name}_Forecast'] = aligned_forecast
                    
                    # Create a download link
                    csv = export_data.to_csv(index=True)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="forecast_vs_actuals.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error creating comparison chart: {str(e)}")
        else:
            st.info("Please enter actual values in the 'Enter Actuals' tab first.")
            
    with tab3:
        st.markdown("#### Accuracy Metrics")
        
        # Debug information - show what we have in session state
        st.write("Debug Information:")
        debug_info = {
            "accuracy_metrics_exists": 'accuracy_metrics' in st.session_state,
            "actuals_series_exists": 'actuals_series' in st.session_state,
            "processed_actuals_flag": st.session_state.get('processed_actuals', False),
            "num_values_entered": len(st.session_state.get('actual_values', {})),
        }
        
        if 'actuals_series' in st.session_state:
            debug_info["actuals_length"] = len(st.session_state.actuals_series)
            
        if 'forecasts' in st.session_state:
            debug_info["forecast_models"] = list(st.session_state.forecasts.keys())
            
        st.json(debug_info)
        
        # Always recalculate metrics when on the metrics tab to ensure we have the latest
        if 'actuals_series' in st.session_state and len(st.session_state.actuals_series) > 0:
            st.warning("Recalculating metrics now...")
            forecasts = st.session_state.forecasts
            actuals = st.session_state.actuals_series
            
            try:
                # Calculate metrics directly
                metrics = {}
                for model_name, forecast_dict in forecasts.items():
                    if 'forecast' in forecast_dict:
                        forecast = forecast_dict['forecast']
                        
                        # Create metrics for this model
                        model_metrics = {}
                        
                        # Show date index information for debugging
                        st.write(f"Forecast dates for {model_name}:")
                        st.write(forecast.index[:5].tolist())
                        st.write("Actual dates:")
                        st.write(actuals.index[:5].tolist())
                        
                        # Use a more flexible approach for date matching
                        # First, convert all dates to normalized date strings
                        forecast_dates = [d.strftime('%Y-%m-%d') for d in forecast.index]
                        actuals_dates = [d.strftime('%Y-%m-%d') for d in actuals.index]
                        
                        # Find common dates using strings
                        common_date_strings = set(forecast_dates).intersection(set(actuals_dates))
                        
                        if common_date_strings:
                            # Convert back to datetime for alignment
                            common_dates = [pd.to_datetime(d) for d in common_date_strings]
                            
                            # Filter forecast and actuals to these dates
                            f_aligned = forecast[forecast.index.isin(common_dates)]
                            a_aligned = actuals[actuals.index.isin(common_dates)]
                            
                            # Always use flexible date matching
                            st.warning(f"Using flexible date matching for {model_name}...")
                            
                            # Create aligned data using nearest dates
                            f_values = []
                            a_values = []
                            aligned_dates = []
                            
                            # For each actual date, find the closest forecast date
                            for a_date, a_value in zip(actuals.index, actuals.values):
                                # Find the nearest forecast date
                                time_diffs = np.array([(a_date - f_date).total_seconds() for f_date in forecast.index])
                                nearest_idx = np.argmin(np.abs(time_diffs))
                                
                                # Get the forecast value for that date
                                nearest_date = forecast.index[nearest_idx]
                                f_value = forecast.iloc[nearest_idx]
                                
                                # Store the aligned values
                                f_values.append(f_value)
                                a_values.append(a_value)
                                aligned_dates.append(a_date)
                                
                            # Create aligned series
                            f_aligned = pd.Series(f_values, index=aligned_dates)
                            a_aligned = pd.Series(a_values, index=aligned_dates)
                            
                            # Show the aligned data
                            st.write("Aligned Data Preview:")
                            alignment_df = pd.DataFrame({
                                'Date': aligned_dates,
                                'Forecast': f_values,
                                'Actual': a_values
                            })
                            st.dataframe(alignment_df)
                            
                            # Calculate metrics
                            mae = np.mean(np.abs(f_aligned.values - a_aligned.values))
                            mse = np.mean((f_aligned.values - a_aligned.values) ** 2)
                            rmse = np.sqrt(mse)
                            
                            if np.all(a_aligned.values != 0):
                                mape = np.mean(np.abs((a_aligned.values - f_aligned.values) / a_aligned.values)) * 100
                            else:
                                mape = 'N/A'
                            
                            model_metrics = {
                                'MAE': round(mae, 2),
                                'RMSE': round(rmse, 2),
                                'MAPE': round(mape, 2) if isinstance(mape, float) else mape,
                                'Samples': len(common_dates)
                            }
                        else:
                            model_metrics = {'error': 'No overlapping data points'}
                        
                        metrics[model_name] = model_metrics
                
                # Store the metrics
                st.session_state.accuracy_metrics = metrics
                st.success("Metrics calculated successfully!")
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
        
        # Display the metrics if available
        if 'accuracy_metrics' in st.session_state and st.session_state.accuracy_metrics:
            metrics = st.session_state.accuracy_metrics
            
            # Create a DataFrame for metrics display
            metrics_data = []
            for model_name, model_metrics in metrics.items():
                if isinstance(model_metrics, dict) and 'error' not in model_metrics:
                    row = {'Model': model_name}
                    row.update(model_metrics)
                    metrics_data.append(row)
            
            if metrics_data:
                st.markdown("### Accuracy Results")
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df)
                
                # Add visualization of metrics
                st.markdown("#### Accuracy Visualization")
                
                # Create bar chart of MAPE or MAE
                metric_to_plot = st.radio("Select Metric to Visualize:", ["MAE", "RMSE"])
                
                # Filter models that have the selected metric
                valid_models = []
                valid_values = []
                
                for model in metrics_data:
                    if metric_to_plot in model and model[metric_to_plot] != 'N/A':
                        valid_models.append(model['Model'])
                        valid_values.append(model[metric_to_plot])
                
                if valid_models:
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=valid_models,
                        y=valid_values,
                        marker_color=['blue', 'green', 'orange', 'red', 'purple'][:len(valid_models)]
                    ))
                    
                    fig.update_layout(
                        title=f"{metric_to_plot} by Model (Lower is Better)",
                        xaxis_title="Model",
                        yaxis_title=metric_to_plot,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No models have valid {metric_to_plot} values.")
            else:
                st.info("No valid metrics data available.")
                st.write("Raw metrics data for debugging:")
                st.json(metrics)
        else:
            st.warning("Please process actual values in the 'Enter Actuals' tab first.")
    
    # Add helpful tips at the bottom
    st.markdown("---")
    st.markdown("""
    #### Tips for Best Results
    
    - Enter values for multiple forecast periods to get more reliable accuracy metrics
    - Compare different models to identify which performs best for your data
    - Look at both MAE and RMSE to understand error magnitude and variability
    - Download comparison data for detailed analysis in Excel or other tools
    """)
