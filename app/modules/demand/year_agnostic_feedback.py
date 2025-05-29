"""
Year-Agnostic Forecast Feedback

A simplified feedback UI that can compare forecasts and actuals regardless of year differences
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import model retraining functionality
from utils.model_retraining import get_enhanced_train_data, save_actuals_to_history


def calculate_metrics(forecasts, actuals):
    """Calculate metrics between forecasts and actuals, ignoring year differences"""
    
    # Create metrics dictionary
    all_metrics = {}
    
    for model_name, forecast_dict in forecasts.items():
        if 'forecast' not in forecast_dict:
            continue
            
        forecast = forecast_dict['forecast']
        
        # Match by month-day regardless of year
        # Extract month and day from both forecast and actuals
        forecast_md = [(d.month, d.day) for d in forecast.index]
        actuals_md = [(d.month, d.day) for d in actuals.index]
        
        # Find matches based on month-day
        matches = []
        for i, (a_month, a_day) in enumerate(actuals_md):
            for j, (f_month, f_day) in enumerate(forecast_md):
                if a_month == f_month and a_day == f_day:
                    matches.append((i, j))
                    
        if not matches:
            # Try month-only matching if no exact matches
            for i, (a_month, _) in enumerate(actuals_md):
                for j, (f_month, _) in enumerate(forecast_md):
                    if a_month == f_month:
                        matches.append((i, j))
        
        # If we have matches, calculate metrics
        if matches:
            a_indices = [m[0] for m in matches]
            f_indices = [m[1] for m in matches]
            
            a_values = actuals.iloc[a_indices].values
            f_values = forecast.iloc[f_indices].values
            
            # Calculate metrics
            mae = np.mean(np.abs(f_values - a_values))
            mse = np.mean((f_values - a_values) ** 2)
            rmse = np.sqrt(mse)
            
            if np.all(a_values != 0):
                mape = np.mean(np.abs((a_values - f_values) / a_values)) * 100
            else:
                mape = float('nan')
            
            all_metrics[model_name] = {
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE': round(mape, 2) if not np.isnan(mape) else 'N/A',
                'Samples': len(matches)
            }
            
            # Also create a dataframe for comparison display
            comparison_data = []
            for a_idx, f_idx in matches:
                comparison_data.append({
                    'Date': actuals.index[a_idx].strftime('%b %d'),
                    'Actual': actuals.iloc[a_idx],
                    f"{model_name} Forecast": forecast.iloc[f_idx]
                })
            
            if model_name not in st.session_state:
                st.session_state[f"{model_name}_comparison"] = pd.DataFrame(comparison_data)
        else:
            all_metrics[model_name] = {'error': 'No matching dates found'}
    
    return all_metrics

def show_year_agnostic_feedback():
    """Display a simplified feedback section that ignores year differences"""
    
    st.markdown("### Forecast Feedback")
    
    # Check if forecasts are available
    if 'forecasts' not in st.session_state:
        st.warning("⚠️ Please generate forecasts first.")
        return
    
    forecasts = st.session_state['forecasts']
    
    # Create tabs
    tab1, tab2 = st.tabs(["Enter Actuals", "View Results"])
    
    with tab1:
        st.markdown("#### Enter Actual Values")
        
        # Initialize session state for actuals
        if 'year_agnostic_actuals' not in st.session_state:
            st.session_state.year_agnostic_actuals = {}
            
        # File upload option
        st.markdown("##### Upload Actual Results File")
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
                
                # Convert to datetime and extract month and day
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Store values by month-day format
                for _, row in df.iterrows():
                    date = row[date_col]
                    month_day = date.strftime('%b %d')
                    st.session_state.year_agnostic_actuals[month_day] = row[value_col]
                
                # Show success message
                st.success(f"✅ Successfully loaded {len(df)} actual values from file.")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                
        st.markdown("##### Or Enter Values Manually")
        
        # Get first forecast model to extract dates
        sample_dates = []
        for model_name, forecast_dict in forecasts.items():
            if 'forecast' in forecast_dict:
                sample_dates = [d.strftime("%b %d") for d in forecast_dict['forecast'].index]
                break
        
        if not sample_dates:
            st.warning("No forecast dates found.")
            return
        
        # Create form for entering values
        st.markdown("##### Enter values for forecast dates")
        
        # Create columns for layout
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Month-day only datelist
            date_option = st.selectbox("Select Date:", sample_dates)
        
        with col2:
            value = st.number_input("Actual Value:", min_value=0.0, step=1.0)
        
        with col3:
            st.write("")
            st.write("")
            if st.button("Add Value"):
                st.session_state.year_agnostic_actuals[date_option] = value
                st.success(f"Added {value} for {date_option}")
        
        # Show current values
        if st.session_state.year_agnostic_actuals:
            st.markdown("#### Current Values")
            
            data = []
            for date_str, val in st.session_state.year_agnostic_actuals.items():
                data.append({"Date": date_str, "Value": val})
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
                
                # Convert to a series for processing
                # Use the current year for dates (doesn't matter since we'll match by month-day)
                current_year = datetime.now().year
                
                dates = []
                values = []
                for date_str, val in st.session_state.year_agnostic_actuals.items():
                    # Parse month and day from "Jan 01" format
                    month_name = date_str.split()[0]
                    day = int(date_str.split()[1])
                    
                    # Convert month name to number
                    datetime_obj = datetime.strptime(f"{month_name} {day}", "%b %d")
                    month = datetime_obj.month
                    
                    # Create a date with the current year
                    date = pd.Timestamp(year=current_year, month=month, day=day)
                    dates.append(date)
                    values.append(val)
                
                if dates:
                    actuals_series = pd.Series(values, index=dates)
                    st.session_state.actuals_series = actuals_series
            
            # Process and Retrain buttons in columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Process button
                if st.button("Process Actuals", key="process_year_agnostic", use_container_width=True):
                    if 'actuals_series' in st.session_state and len(st.session_state.actuals_series) > 0:
                        # Calculate metrics
                        metrics = calculate_metrics(forecasts, st.session_state.actuals_series)
                        st.session_state.year_agnostic_metrics = metrics
                        
                        # Save actuals to history directory
                        save_actuals_to_history(st.session_state.actuals_series, history_dir="forecast_history")
                        
                        st.success("✅ Processed successfully!")
                        st.info("Go to the 'View Results' tab to see metrics")
                    else:
                        st.error("No actual values to process")
            
            with col2:
                # Retrain button with a different color
                if st.button("↻ ENHANCE MODELS WITH ACTUALS", key="retrain_models", use_container_width=True, type="primary"):
                    if 'actuals_series' in st.session_state and len(st.session_state.actuals_series) > 0:
                        # Get original training data from forecasts
                        training_data = None
                        for key in st.session_state.keys():
                            if key.endswith('_training_data') and isinstance(st.session_state[key], pd.Series):
                                training_data = st.session_state[key]
                                break
                        
                        if training_data is None and 'training_data' in st.session_state:
                            training_data = st.session_state.training_data
                            
                        if training_data is not None:
                            # Store enhanced training data in session state
                            enhanced_training_data = get_enhanced_train_data(
                                st.session_state.actuals_series, 
                                training_data
                            )
                            
                            # Store the enhanced data
                            st.session_state.enhanced_training_data = enhanced_training_data
                            
                            # Set a flag to use enhanced data for next forecast
                            st.session_state.use_enhanced_data = True
                            
                            st.success("✅ Models enhanced with actual data!")
                            st.info("Next time you forecast, the models will use your actual data to improve results.")
                        else:
                            st.error("Original training data not found. Can't enhance models.")
                            st.info("Try generating a forecast first, then enter actuals, then enhance.")
                    else:
                        st.error("No actual values to use for enhancement")
            
            # Explain the difference between the two buttons
            st.markdown("""
            - **Process Actuals**: Calculates metrics but doesn't modify models
            - **Enhance Models**: Updates models with your actual data for better future forecasts
            """)
            
            # Clear button
            if st.button("Clear Values"):
                st.session_state.year_agnostic_actuals = {}
                if 'actuals_series' in st.session_state:
                    del st.session_state.actuals_series
                st.experimental_rerun()
    
    with tab2:
        st.markdown("#### Results")
        
        if 'year_agnostic_metrics' in st.session_state:
            metrics = st.session_state.year_agnostic_metrics
            
            # Create DataFrame for metrics display
            metrics_data = []
            for model_name, model_metrics in metrics.items():
                if isinstance(model_metrics, dict) and 'error' not in model_metrics:
                    row = {'Model': model_name}
                    row.update(model_metrics)
                    metrics_data.append(row)
            
            if metrics_data:
                st.markdown("### Accuracy Metrics")
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df)
                
                # Show comparisons for each model
                st.markdown("### Forecast vs Actual Comparisons")
                
                for model_name in metrics:
                    if f"{model_name}_comparison" in st.session_state:
                        st.markdown(f"#### {model_name}")
                        st.dataframe(st.session_state[f"{model_name}_comparison"])
                        
                        # Create comparison chart
                        comparison_df = st.session_state[f"{model_name}_comparison"]
                        
                        fig = go.Figure()
                        
                        # Add actual values
                        fig.add_trace(go.Scatter(
                            x=comparison_df['Date'],
                            y=comparison_df['Actual'],
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=comparison_df['Date'],
                            y=comparison_df[f"{model_name} Forecast"],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='blue')
                        ))
                        
                        fig.update_layout(
                            title=f"{model_name}: Forecast vs Actual",
                            xaxis_title="Date",
                            yaxis_title="Value",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid metrics available")
        else:
            st.info("Please process actual values in the 'Enter Actuals' tab first")
