"""
Integrated Business Planning (IBP) UI Module
This module provides the Streamlit user interface for the IBP application,
focusing on Demand Planning and Inventory Optimization.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from utils.forecasting import *  # Import forecasting utilities
from modules.demand.month_detection import detect_month_names, month_to_date

# Page configuration is handled in main.py
# Do not set page config here

def show_demand_planning():
    """Main function to display the Demand Planning module UI"""
    st.title("Demand Planning")
    
    # Sidebar for navigation and inputs
    with st.sidebar:
        st.header("Forecast Settings")
        forecast_period = st.slider("Forecast Periods", 1, 24, 12)
        
        # Model selection
        st.subheader("Models to Run")
        models = {
            "ARIMA": st.checkbox("ARIMA", value=True),
            "Exponential Smoothing": st.checkbox("Exponential Smoothing", value=True),
            "Prophet": st.checkbox("Prophet", value=False),
            "XGBoost": st.checkbox("XGBoost", value=False),
            "Ensemble": st.checkbox("Ensemble", value=True)
        }
        
        # Add upload button
        uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Display data overview
            st.subheader("Data Overview")
            st.dataframe(data.head())
            
            # Data preprocessing
            st.subheader("Data Preprocessing")
            
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Date Column", data.columns)
            with col2:
                target_col = st.selectbox("Target Column", data.columns)
            
            # Convert to datetime and set as index
            if st.button("Prepare Data"):
                # Convert date column to datetime
                data[date_col] = pd.to_datetime(data[date_col])
                data.set_index(date_col, inplace=True)
                
                # Store in session state
                st.session_state['data'] = data
                st.session_state['target_col'] = target_col
                
                st.success("Data prepared successfully!")
                
                # Display time series plot
                fig = px.line(data, y=target_col, title="Historical Data")
                st.plotly_chart(fig, use_container_width=True)
                
                # Run forecasting models
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecasts..."):
                        # Train-test split
                        train_size = int(len(data) * 0.8)
                        train_data = data.iloc[:train_size]
                        test_data = data.iloc[train_size:]
                        
                        # Prepare future dates
                        last_date = data.index[-1]
                        future_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=forecast_period,
                            freq='D'
                        )
                        
                        # Store forecasts
                        forecasts = {}
                        
                        # Generate forecasts based on selected models
                        if models["ARIMA"]:
                            with st.spinner("Running ARIMA model..."):
                                try:
                                    model = ARIMA(train_data[target_col], order=(1,1,1))
                                    fitted = model.fit()
                                    arima_forecast = fitted.forecast(steps=forecast_period)
                                    arima_forecast.index = future_dates[:len(arima_forecast)]
                                    forecasts["ARIMA"] = arima_forecast
                                    st.success("ARIMA forecast complete")
                                except Exception as e:
                                    st.error(f"ARIMA error: {e}")
                        
                        if models["Exponential Smoothing"]:
                            with st.spinner("Running Exponential Smoothing model..."):
                                try:
                                    model = ExponentialSmoothing(
                                        train_data[target_col],
                                        seasonal_periods=12,
                                        trend='add',
                                        seasonal='add'
                                    )
                                    fitted = model.fit()
                                    es_forecast = fitted.forecast(forecast_period)
                                    es_forecast.index = future_dates[:len(es_forecast)]
                                    forecasts["Exponential Smoothing"] = es_forecast
                                    st.success("Exponential Smoothing forecast complete")
                                except Exception as e:
                                    st.error(f"Exponential Smoothing error: {e}")
                        
                        if models["Prophet"]:
                            with st.spinner("Running Prophet model..."):
                                try:
                                    from prophet import Prophet
                                    # Prepare data for Prophet
                                    prophet_data = pd.DataFrame({
                                        'ds': train_data.index,
                                        'y': train_data[target_col].values
                                    })
                                    model = Prophet()
                                    model.fit(prophet_data)
                                    future = pd.DataFrame({'ds': future_dates})
                                    forecast = model.predict(future)
                                    prophet_forecast = pd.Series(
                                        forecast['yhat'].values, 
                                        index=future_dates
                                    )
                                    forecasts["Prophet"] = prophet_forecast
                                    st.success("Prophet forecast complete")
                                except Exception as e:
                                    st.error(f"Prophet error: {e}")
                        
                        if models["XGBoost"]:
                            with st.spinner("Running XGBoost model..."):
                                try:
                                    import xgboost as xgb
                                    # Feature engineering for time series
                                    df = train_data.copy()
                                    df['lag1'] = df[target_col].shift(1)
                                    df['lag2'] = df[target_col].shift(2)
                                    df['lag3'] = df[target_col].shift(3)
                                    df.dropna(inplace=True)
                                    
                                    # Prepare train data
                                    X = df[['lag1', 'lag2', 'lag3']]
                                    y = df[target_col]
                                    
                                    # Train model
                                    model = xgb.XGBRegressor(objective='reg:squarederror')
                                    model.fit(X, y)
                                    
                                    # Create future features for prediction
                                    last_values = train_data[target_col].iloc[-3:].values
                                    xgb_forecast = []
                                    
                                    for i in range(forecast_period):
                                        features = np.array([last_values[i % 3], last_values[(i+1) % 3], last_values[(i+2) % 3]]).reshape(1, -1)
                                        pred = model.predict(features)[0]
                                        xgb_forecast.append(pred)
                                        last_values = np.append(last_values[1:], pred)
                                    
                                    forecasts["XGBoost"] = pd.Series(xgb_forecast, index=future_dates)
                                    st.success("XGBoost forecast complete")
                                except Exception as e:
                                    st.error(f"XGBoost error: {e}")
                        
                        # Create ensemble forecast if multiple forecasts are available and ensemble is selected
                        if models["Ensemble"] and len(forecasts) > 1:
                            with st.spinner("Creating ensemble forecast..."):
                                try:
                                    # Simple average ensemble
                                    ensemble_data = pd.DataFrame({k: v for k, v in forecasts.items()})
                                    ensemble_forecast = ensemble_data.mean(axis=1)
                                    forecasts["Ensemble"] = ensemble_forecast
                                    st.success("Ensemble forecast complete")
                                except Exception as e:
                                    st.error(f"Ensemble error: {e}")
                        
                        # Display forecast results
                        if forecasts:
                            st.session_state['forecasts'] = forecasts
                            display_forecast_results(train_data, forecasts, target_col)
                        else:
                            st.warning("No forecasts were generated. Please select at least one model.")
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("Please upload a data file to begin.")

def show_inventory_optimization():
    """Main function to display the Inventory Optimization module UI"""
    st.title("Inventory Optimization")
    
    # Sidebar for inventory optimization settings
    with st.sidebar:
        st.header("Optimization Parameters")
        service_level = st.slider("Service Level (%)", 80, 99, 95)
        lead_time = st.number_input("Lead Time (days)", 1, 100, 14)
        holding_cost = st.slider("Holding Cost (%)", 10, 50, 25)
    
    # Main content area
    st.info("Inventory optimization helps you determine optimal inventory levels, reorder points, and safety stock based on your demand forecast and business constraints.")
    
    # Check if forecast data is available
    if 'forecasts' in st.session_state and 'data' in st.session_state:
        # Display available forecast models
        forecast_models = list(st.session_state['forecasts'].keys())
        selected_model = st.selectbox("Select Forecast Model", forecast_models)
        
        if st.button("Run Optimization"):
            with st.spinner("Optimizing inventory levels..."):
                # Get the selected forecast
                forecast = st.session_state['forecasts'][selected_model]
                
                # Run optimization
                optimization_results = optimize_inventory(
                    inventory_data=forecast,
                    service_level=service_level/100,  # Convert to decimal
                    lead_time=lead_time,
                    holding_cost=holding_cost/100  # Convert to decimal
                )
                
                # Display results
                display_optimization_results(optimization_results)
    else:
        st.warning("Please generate demand forecasts first in the Demand Planning module.")

def display_forecast_results(historical_data, forecast_results, target_col):
    """Display the forecast results using plots and metrics"""
    st.subheader("Forecast Results")
    
    # Create plot with historical and forecast data
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data[target_col],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add each forecast
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (model_name, forecast) in enumerate(forecast_results.items()):
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
        title='Demand Forecast',
        xaxis_title='Date',
        yaxis_title=target_col,
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display metrics if we have test data
    if len(historical_data) > int(len(historical_data) * 0.8):
        st.subheader("Forecast Metrics")
        
        # Split data into train/test for evaluation
        train_size = int(len(historical_data) * 0.8)
        test_data = historical_data.iloc[train_size:]
        
        # Find overlapping dates between forecasts and test data
        metrics_data = []
        
        for model_name, forecast in forecast_results.items():
            common_dates = test_data.index.intersection(forecast.index)
            
            if len(common_dates) > 0:
                actual = test_data.loc[common_dates, target_col]
                pred = forecast.loc[common_dates]
                
                rmse = np.sqrt(mean_squared_error(actual, pred))
                mae = mean_absolute_error(actual, pred)
                
                # Calculate MAPE with protection against zero values
                if np.any(actual == 0):
                    mape = np.mean(np.abs((actual - pred) / (actual + 1e-5))) * 100
                else:
                    mape = np.mean(np.abs((actual - pred) / actual)) * 100
                
                metrics_data.append({
                    'Model': model_name,
                    'RMSE': round(rmse, 2),
                    'MAE': round(mae, 2),
                    'MAPE (%)': round(mape, 2)
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df)
        else:
            st.info("No common dates between forecasts and test data for evaluation.")

def optimize_inventory(inventory_data, service_level, lead_time, holding_cost):
    """Optimize inventory levels based on parameters"""
    # Calculate standard deviation of demand
    demand_std = inventory_data.std()
    
    # Calculate average demand
    demand_avg = inventory_data.mean()
    
    # Calculate safety factor based on service level
    # Using an approximation of the inverse normal CDF
    if service_level >= 0.5:
        safety_factor = 0.84 + (service_level - 0.5) * 3
    else:
        safety_factor = (service_level - 0.5) * 3
    
    # Calculate safety stock
    safety_stock = safety_factor * demand_std * np.sqrt(lead_time)
    
    # Calculate reorder point
    reorder_point = demand_avg * lead_time + safety_stock
    
    # Calculate economic order quantity (EOQ)
    annual_demand = demand_avg * 365
    ordering_cost = 100  # Assumed fixed ordering cost
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost * demand_avg))
    
    # Calculate inventory cost components
    holding_cost_value = holding_cost * (eoq/2 + safety_stock) * demand_avg
    ordering_cost_value = (annual_demand / eoq) * ordering_cost
    total_cost = holding_cost_value + ordering_cost_value
    
    # Prepare results
    results = {
        'reorder_points': reorder_point,
        'safety_stock': safety_stock,
        'economic_order_quantity': eoq,
        'cost_breakdown': {
            'holding_cost': holding_cost_value,
            'ordering_cost': ordering_cost_value,
            'total_cost': total_cost
        }
    }
    
    return results

def display_optimization_results(results):
    """Display inventory optimization results"""
    st.subheader("Inventory Optimization Results")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Reorder Point", f"{results['reorder_points']:.2f}")
        st.caption("When inventory reaches this level, place a new order")
    
    with col2:
        st.metric("Safety Stock", f"{results['safety_stock']:.2f}")
        st.caption("Extra inventory to prevent stockouts")
    
    with col3:
        st.metric("Economic Order Quantity", f"{results['economic_order_quantity']:.2f}")
        st.caption("Optimal order size to minimize costs")
    
    # Display cost breakdown
    st.subheader("Cost Breakdown")
    costs = results['cost_breakdown']
    
    # Create bar chart for costs
    cost_data = pd.DataFrame({
        'Cost Type': ['Holding Cost', 'Ordering Cost', 'Total Cost'],
        'Amount': [costs['holding_cost'], costs['ordering_cost'], costs['total_cost']]
    })
    
    fig = px.bar(
        cost_data, 
        x='Cost Type', 
        y='Amount',
        title='Inventory Cost Breakdown',
        color='Cost Type',
        text_auto='.2f'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional explanation
    st.info("""
    **Inventory Optimization Explanation:**
    - **Reorder Point**: The inventory level at which you should place a new order
    - **Safety Stock**: Buffer inventory to protect against variability in demand and lead time
    - **Economic Order Quantity (EOQ)**: The optimal order quantity that minimizes total inventory costs
    """)

# Main entry point
if __name__ == "__main__":
    st.sidebar.title("IBP Navigation")
    app_mode = st.sidebar.selectbox("Select Module", ["IBP for Demand", "IBP for Inventory"])
    
    if app_mode == "IBP for Demand":
        show_demand_planning()
    elif app_mode == "IBP for Inventory":
        show_inventory_optimization()
