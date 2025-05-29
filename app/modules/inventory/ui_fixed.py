"""
Inventory Optimization module UI components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
from io import StringIO
import math

# Ensure app directory is in path
app_path = Path(__file__).parent.parent.parent
sys.path.append(str(app_path))

# Import config
import config

def calculate_eoq(annual_demand, order_cost, holding_cost):
    """Calculate Economic Order Quantity."""
    return math.sqrt((2 * annual_demand * order_cost) / holding_cost)

def calculate_reorder_point(lead_time_days, daily_demand, safety_stock):
    """Calculate Reorder Point."""
    return (lead_time_days * daily_demand) + safety_stock

def calculate_safety_stock(service_level, lead_time_days, std_dev_demand):
    """Calculate Safety Stock based on service level."""
    # Convert service level to z-score (normal distribution)
    z_scores = {
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33,
    }
    
    # Get closest z-score
    closest_service_level = min(z_scores.keys(), key=lambda x: abs(x - service_level))
    z_score = z_scores[closest_service_level]
    
    # Calculate safety stock
    return z_score * std_dev_demand * math.sqrt(lead_time_days)

def generate_sample_inventory_data():
    """Generate sample inventory data for demonstration."""
    # Create sample product data
    products = [
        {"product_id": "P001", "name": "Premium Widget", "category": "A", "unit_cost": 25.0},
        {"product_id": "P002", "name": "Standard Widget", "category": "B", "unit_cost": 15.0},
        {"product_id": "P003", "name": "Economy Widget", "category": "C", "unit_cost": 8.0},
        {"product_id": "P004", "name": "Deluxe Gadget", "category": "A", "unit_cost": 45.0},
        {"product_id": "P005", "name": "Basic Gadget", "category": "B", "unit_cost": 20.0},
    ]
    
    # Create inventory data
    inventory_data = []
    
    for product in products:
        # Generate random but realistic inventory metrics
        annual_demand = np.random.randint(1000, 10000)
        lead_time_days = np.random.randint(7, 45)
        order_cost = np.random.randint(50, 200)
        holding_cost_pct = np.random.uniform(0.1, 0.3)
        holding_cost = product['unit_cost'] * holding_cost_pct
        current_stock = np.random.randint(100, 1000)
        std_dev_demand = annual_demand * np.random.uniform(0.05, 0.2) / np.sqrt(365)
        
        # Calculate daily demand
        daily_demand = annual_demand / 365
        
        # Add to inventory data
        inventory_data.append({
            "product_id": product['product_id'],
            "name": product['name'],
            "category": product['category'],
            "unit_cost": product['unit_cost'],
            "annual_demand": annual_demand,
            "daily_demand": daily_demand,
            "lead_time_days": lead_time_days,
            "order_cost": order_cost,
            "holding_cost_pct": holding_cost_pct,
            "holding_cost": holding_cost,
            "current_stock": current_stock,
            "std_dev_demand": std_dev_demand,
        })
    
    # Create DataFrame
    df = pd.DataFrame(inventory_data)
    
    return df

def show_inventory_optimization():
    """
    Show inventory optimization UI components.
    """
    # Page title
    st.markdown("# 📦 Inventory Optimization")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Inventory Data", "EOQ Analysis", "Replenishment Planning"])
    
    # Generate sample data if needed
    sample_data_path = os.path.join(app_path, "data", "sample_inventory_data.csv")
    if not os.path.exists(sample_data_path):
        os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
        
        # Generate and save sample data
        sample_df = generate_sample_inventory_data()
        sample_df.to_csv(sample_data_path, index=False)
    
    # Load sample data
    sample_df = pd.read_csv(sample_data_path)
    
    # Inventory Data Tab
    with tab1:
        st.markdown("### Inventory Data")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source:",
            ["Upload CSV", "Sample Data", "Database Connection"],
            horizontal=True
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your inventory data (CSV):", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['inventory_data'] = df
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            else:
                st.info("Please upload a CSV file with inventory data.")
                
        elif data_source == "Sample Data":
            st.info("Using sample inventory data for demonstration.")
            st.session_state['inventory_data'] = sample_df
            
        elif data_source == "Database Connection":
            st.info("Database connection feature will be available in the future.")
            st.session_state['inventory_data'] = sample_df  # Use sample data for now
        
        # Display data preview
        if 'inventory_data' in st.session_state:
            st.markdown("#### Data Preview")
            st.dataframe(st.session_state['inventory_data'])
            
            # ABC Analysis
            st.markdown("#### ABC Analysis")
            
            if st.button("Run ABC Analysis"):
                # Get data
                df = st.session_state['inventory_data'].copy()
                
                # Calculate annual value
                df['annual_value'] = df['annual_demand'] * df['unit_cost']
                
                # Sort by annual value
                df = df.sort_values('annual_value', ascending=False)
                
                # Calculate cumulative percentages
                total_value = df['annual_value'].sum()
                df['percentage'] = (df['annual_value'] / total_value) * 100
                df['cumulative_percentage'] = df['percentage'].cumsum()
                
                # Classify products
                df['class'] = 'C'
                df.loc[df['cumulative_percentage'] <= 80, 'class'] = 'A'
                df.loc[(df['cumulative_percentage'] > 80) & (df['cumulative_percentage'] <= 95), 'class'] = 'B'
                
                # Display results
                st.dataframe(df[['product_id', 'name', 'annual_value', 'percentage', 'cumulative_percentage', 'class']])
                
                # Create Pareto chart
                fig = go.Figure()
                
                # Add annual value bars
                fig.add_trace(go.Bar(
                    x=df['name'],
                    y=df['annual_value'],
                    name='Annual Value',
                    marker_color='blue'
                ))
                
                # Add cumulative percentage line
                fig.add_trace(go.Scatter(
                    x=df['name'],
                    y=df['cumulative_percentage'],
                    name='Cumulative %',
                    marker_color='red',
                    mode='lines+markers',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Pareto Analysis of Inventory Value',
                    xaxis_title='Product',
                    yaxis_title='Annual Value',
                    yaxis2=dict(
                        title='Cumulative %',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    ),
                    legend=dict(x=0.01, y=0.99),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # EOQ Analysis Tab
    with tab2:
        st.markdown("### Economic Order Quantity (EOQ) Analysis")
        
        if 'inventory_data' not in st.session_state:
            st.info("Please load inventory data in the Inventory Data tab first.")
        else:
            # Product selection
            st.markdown("#### Select Product")
            
            # Get inventory data
            df = st.session_state['inventory_data']
            
            # Product selection - handle different column name possibilities
            product_name_column = None
            
            # Try to identify product name column
            possible_name_columns = ['name', 'product_name', 'item_name', 'description', 'product_id']
            for col in possible_name_columns:
                if col in df.columns:
                    product_name_column = col
                    break
                    
            if product_name_column is None:
                # If no product name column found, use the first column as identifier
                product_name_column = df.columns[0]
                st.warning(f"No product name column found. Using '{product_name_column}' column as product identifier.")
            
            # Display product selection dropdown
            selected_product = st.selectbox(
                "Select Product:",
                df[product_name_column].tolist()
            )
            
            # Get selected product data
            product_data = df[df[product_name_column] == selected_product].iloc[0]
            
            # Display product details
            st.markdown("#### Product Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Demand", f"{product_data['annual_demand']:,.0f} units")
                
            with col2:
                st.metric("Unit Cost", f"${product_data['unit_cost']:.2f}")
                
            with col3:
                st.metric("Lead Time", f"{product_data['lead_time_days']} days")
            
            # EOQ Parameters
            st.markdown("#### EOQ Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                annual_demand = st.number_input(
                    "Annual Demand (units):",
                    min_value=1.0,
                    value=float(product_data['annual_demand']),
                    step=100.0
                )
                
                order_cost = st.number_input(
                    "Order Cost ($):",
                    min_value=1.0,
                    value=float(product_data['order_cost']),
                    step=10.0
                )
                
            with col2:
                holding_cost_pct = st.slider(
                    "Holding Cost (% of unit cost):",
                    min_value=1.0,
                    max_value=50.0,
                    value=float(product_data['holding_cost_pct'] * 100),
                    step=1.0
                ) / 100
                
                unit_cost = st.number_input(
                    "Unit Cost ($):",
                    min_value=0.01,
                    value=float(product_data['unit_cost']),
                    step=1.0
                )
            
            # Calculate EOQ
            holding_cost = unit_cost * holding_cost_pct
            eoq = calculate_eoq(annual_demand, order_cost, holding_cost)
            annual_orders = annual_demand / eoq
            order_cycle_days = 365 / annual_orders
            
            # Display EOQ results
            st.markdown("#### EOQ Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Economic Order Quantity", f"{eoq:.0f} units")
                
            with col2:
                st.metric("Annual Orders", f"{annual_orders:.1f} orders")
                
            with col3:
                st.metric("Order Cycle", f"{order_cycle_days:.0f} days")
            
            # Total Annual Cost
            annual_order_cost = annual_orders * order_cost
            annual_holding_cost = (eoq / 2) * holding_cost
            total_annual_cost = annual_order_cost + annual_holding_cost
            
            st.markdown("#### Annual Cost Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Order Cost", f"${annual_order_cost:.2f}")
                
            with col2:
                st.metric("Annual Holding Cost", f"${annual_holding_cost:.2f}")
                
            with col3:
                st.metric("Total Annual Cost", f"${total_annual_cost:.2f}")
    
    # Replenishment Planning Tab
    with tab3:
        st.markdown("### Replenishment Planning")
        
        if 'inventory_data' not in st.session_state:
            st.info("Please load inventory data in the Inventory Data tab first.")
        else:
            # Get inventory data
            df = st.session_state['inventory_data']
            
            # Product selection - handle different column name possibilities
            product_name_column = None
            
            # Try to identify product name column
            possible_name_columns = ['name', 'product_name', 'item_name', 'description', 'product_id']
            for col in possible_name_columns:
                if col in df.columns:
                    product_name_column = col
                    break
                    
            if product_name_column is None:
                # If no product name column found, use the first column as identifier
                product_name_column = df.columns[0]
                st.warning(f"No product name column found. Using '{product_name_column}' column as product identifier.")
            
            # Display product selection dropdown
            selected_product = st.selectbox(
                "Select Product:",
                df[product_name_column].tolist()
            )
            
            # Get selected product data
            product_data = df[df[product_name_column] == selected_product].iloc[0]
            
            # Service level selection
            service_level = st.slider(
                "Service Level (%):",
                min_value=90,
                max_value=99,
                value=95,
                step=1
            ) / 100
            
            # Calculate safety stock
            safety_stock = calculate_safety_stock(
                service_level,
                product_data['lead_time_days'],
                product_data['std_dev_demand']
            )
            
            # Calculate reorder point
            reorder_point = calculate_reorder_point(
                product_data['lead_time_days'],
                product_data['daily_demand'],
                safety_stock
            )
            
            # Display results
            st.markdown("#### Replenishment Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Safety Stock", f"{safety_stock:.0f} units")
                
            with col2:
                st.metric("Reorder Point", f"{reorder_point:.0f} units")
                
            with col3:
                st.metric("Current Stock", f"{product_data['current_stock']:.0f} units")
            
            # Stock status
            stock_status = "OK"
            days_to_stockout = 0
            
            if product_data['current_stock'] <= reorder_point:
                if product_data['current_stock'] <= safety_stock:
                    stock_status = "CRITICAL"
                else:
                    stock_status = "ORDER NOW"
                days_to_stockout = product_data['current_stock'] / product_data['daily_demand']
            
            # Display stock status
            st.markdown("#### Stock Status")
            
            col1, col2 = st.columns(2)
            
            with col1:
                status_color = {
                    "OK": "green",
                    "ORDER NOW": "orange",
                    "CRITICAL": "red"
                }
                
                st.markdown(
                    f"<h3 style='color: {status_color[stock_status]};'>{stock_status}</h3>",
                    unsafe_allow_html=True
                )
                
            with col2:
                if stock_status != "OK":
                    st.metric(
                        "Days Until Stockout",
                        f"{days_to_stockout:.0f} days"
                    )
            
            # Calculate optimal order quantity
            if stock_status != "OK":
                # Calculate EOQ
                holding_cost = product_data['unit_cost'] * product_data['holding_cost_pct']
                eoq = calculate_eoq(
                    product_data['annual_demand'],
                    product_data['order_cost'],
                    holding_cost
                )
                
                st.markdown("#### Recommended Order")
                st.info(f"Place an order for {eoq:.0f} units as soon as possible.")
