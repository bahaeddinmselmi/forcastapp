"""
S&OP Alignment module UI components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import Excel detector
from utils.excel_detector import load_excel_with_smart_detection
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json

# Ensure app directory is in path
app_path = Path(__file__).parent.parent.parent
sys.path.append(str(app_path))

# Import config
import config

def generate_sample_sop_data():
    """Generate sample S&OP data for demonstration."""
    # Create date range for next 12 months
    start_date = datetime.now().replace(day=1)
    dates = [start_date + timedelta(days=30*i) for i in range(12)]
    
    # Create products
    products = [
        {"product_id": "P001", "name": "Premium Widget", "category": "A", "unit_price": 50.0, "unit_cost": 25.0},
        {"product_id": "P002", "name": "Standard Widget", "category": "B", "unit_price": 30.0, "unit_cost": 15.0},
        {"product_id": "P003", "name": "Economy Widget", "category": "C", "unit_price": 15.0, "unit_cost": 8.0},
        {"product_id": "P004", "name": "Deluxe Gadget", "category": "A", "unit_price": 80.0, "unit_cost": 45.0},
        {"product_id": "P005", "name": "Basic Gadget", "category": "B", "unit_price": 40.0, "unit_cost": 20.0},
    ]
    
    # Create S&OP data
    sop_data = []
    
    for product in products:
        base_demand = np.random.randint(100, 500)
        base_supply = base_demand  # Initially set supply equal to demand
        
        for i, date in enumerate(dates):
            # Add some seasonality and trend
            seasonal_factor = 1.0 + 0.2 * np.sin(np.pi * i / 6)
            trend_factor = 1.0 + 0.03 * i
            
            # Calculate demand for this month
            monthly_demand = int(base_demand * seasonal_factor * trend_factor)
            
            # Supply might differ from demand
            supply_variation = np.random.uniform(0.9, 1.1)
            monthly_supply = int(monthly_demand * supply_variation)
            
            # Calculate financial metrics
            revenue = monthly_demand * product['unit_price']
            cogs = monthly_supply * product['unit_cost']
            gross_margin = revenue - cogs
            gross_margin_pct = (gross_margin / revenue) * 100 if revenue > 0 else 0
            
            # Inventory calculation
            beginning_inventory = 0 if i == 0 else sop_data[-1]['ending_inventory']
            ending_inventory = beginning_inventory + monthly_supply - monthly_demand
            inventory_value = ending_inventory * product['unit_cost']
            
            sop_data.append({
                "product_id": product['product_id'],
                "product_name": product['name'],
                "category": product['category'],
                "date": date.strftime("%Y-%m-%d"),
                "forecast_demand": monthly_demand,
                "planned_supply": monthly_supply,
                "revenue": revenue,
                "cogs": cogs,
                "gross_margin": gross_margin,
                "gross_margin_pct": gross_margin_pct,
                "beginning_inventory": beginning_inventory,
                "ending_inventory": ending_inventory,
                "inventory_value": inventory_value,
                "unit_price": product['unit_price'],
                "unit_cost": product['unit_cost']
            })
    
    return pd.DataFrame(sop_data)

def show_sop_alignment():
    """
    Display the S&OP Alignment module interface.
    """
    st.markdown("## S&OP Alignment")
    
    st.markdown("""
    The IBP for S&OP module helps you align supply and demand planning with financial goals.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "S&OP Data", 
        "Demand-Supply Balance", 
        "Financial Impact",
        "Executive Dashboard"
    ])
    
    # Create sample data directory if it doesn't exist
    data_dir = os.path.join(app_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate and save sample data if it doesn't exist
    sample_data_path = os.path.join(data_dir, "sample_sop_data.csv")
    if not os.path.exists(sample_data_path):
        sample_df = generate_sample_sop_data()
        sample_df.to_csv(sample_data_path, index=False)
    
    # Load sample data
    sample_df = pd.read_csv(sample_data_path)
    
    # S&OP Data Tab
    with tab1:
        st.markdown("### S&OP Data")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source:",
            ["Upload Data", "Load From Directory", "Sample Data", "Database Connection"],
            horizontal=True
        )
        
        if data_source == "Upload Data":
            uploaded_file = st.file_uploader("Upload your S&OP data:", type=["csv", "xlsx", "xls"])
            if uploaded_file is not None:
                try:
                    # Determine file type and read accordingly
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_type in ['xlsx', 'xls']:
                        # Create a container for AI detection or manual override
                        detection_container = st.container()
                        
                        # Add a manual override option
                        with st.expander("Manual Excel Import Options (If AI detection fails)", expanded=False):
                            st.info("If the AI doesn't correctly detect your data, use these manual options to specify exactly where your data is located.")
                            
                            # Get available sheet names
                            excel = pd.ExcelFile(uploaded_file)
                            sheet_names = excel.sheet_names
                            
                            # Let user select which sheet to use
                            selected_sheet = st.selectbox(
                                "Select Sheet", 
                                options=sheet_names,
                                key="sop_manual_sheet"
                            )
                            
                            # Options for data range
                            col1, col2 = st.columns(2)
                            with col1:
                                start_row = st.number_input("Start Row (0-based)", min_value=0, value=0, key="sop_start_row")
                                has_header = st.checkbox("First row is header", value=True, key="sop_manual_header")
                            
                            with col2:
                                end_row = st.number_input("End Row (leave at 1000 to read all)", min_value=1, value=1000, key="sop_end_row")
                                first_col = st.text_input("First Column (e.g., A)", value="A", key="sop_first_col")
                                last_col = st.text_input("Last Column (e.g., Z or leave empty for all)", value="", key="sop_last_col")
                            
                            # Convert Excel column letters to usecols parameter
                            def get_usecols_param(first, last):
                                if not first:
                                    return None
                                if not last:
                                    return first + ":" if first else None
                                return f"{first}:{last}"
                            
                            usecols = get_usecols_param(first_col, last_col)
                            
                            # Preview button with manual settings
                            if st.button("Preview with Manual Settings", key="sop_manual_preview"):
                                try:
                                    # Calculate actual skiprows
                                    skiprows = list(range(start_row))
                                    
                                    # Calculate nrows (rows to read)
                                    nrows = end_row - start_row if end_row > start_row else None
                                    
                                    # Read with manual settings
                                    preview_df = pd.read_excel(
                                        uploaded_file,
                                        sheet_name=selected_sheet,
                                        header=0 if has_header else None,
                                        skiprows=skiprows,
                                        nrows=nrows,
                                        usecols=usecols
                                    )
                                    st.write(f"Preview with manual settings ({len(preview_df)} rows):")
                                    st.dataframe(preview_df.head(15))
                                    
                                    # Option to use these settings
                                    if st.button("Use These Settings", key="sop_use_manual"):
                                        df = preview_df
                                        st.session_state['sop_override_active'] = True
                                        st.success("Using manual settings for data import!")
                                except Exception as e:
                                    st.error(f"Error with manual settings: {e}")
                        
                        # Only use AI detection if manual override is not active
                        if not st.session_state.get('sop_override_active', False):
                            # Use AI-based table detection for Excel files
                            with detection_container:
                                with st.spinner("AI is analyzing your Excel file to automatically detect data tables..."):
                                    # Show a progress message
                                    progress_placeholder = st.empty()
                                    progress_placeholder.info("🔍 Detecting data tables in your Excel file...")
                                    
                                    # Use the AI detector to load the Excel file
                                    df = load_excel_with_smart_detection(uploaded_file)
                                    
                                    # Show success message with detected table info
                                    rows, cols = df.shape
                                    progress_placeholder.success(f"✅ Successfully detected data table with {rows} rows and {cols} columns!")
                            
                            # Show preview of detected data
                            with st.expander("Preview of detected data", expanded=True):
                                st.write(f"Showing first 15 rows of {len(df)} total rows:")
                                st.dataframe(df.head(15))  # Show more rows in preview
                                
                                # Add option to see more rows if needed
                                if len(df) > 15 and st.button("Show more rows", key="sop_more_rows"):
                                    st.dataframe(df)
                    else:  # Default to CSV
                        df = pd.read_csv(uploaded_file)
                    
                    st.session_state['sop_data'] = df
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            else:
                st.info("Please upload your S&OP data file (Excel or CSV).")
                st.caption("Our AI will automatically detect data tables in your file, even if they're not at the top of the sheet.")
                
        elif data_source == "Load From Directory":
            st.info("Loading data from the app/data/sample_files directory")
            file_path = os.path.join(app_path, "data", "sample_files", "sop_data.xlsx")
            
            if os.path.exists(file_path):
                try:
                    # Use our smart Excel detector to load the file
                    with st.spinner("Loading your Excel file from directory..."):
                        df = load_excel_with_smart_detection(file_path)
                        rows, cols = df.shape
                        st.success(f"✅ Successfully loaded data file with {rows} rows and {cols} columns!")
                        
                        # Show preview of detected data
                        with st.expander("Preview of detected data", expanded=True):
                            st.write(f"Showing first 15 rows of {len(df)} total rows:")
                            st.dataframe(df.head(15))
                            
                            # Add option to see more rows if needed
                            if len(df) > 15 and st.button("Show more rows", key="sop_dir_more_rows"):
                                st.dataframe(df)
                                
                        # Check if data might have month-only dates (without years)
                        date_column = None
                        has_month_names = False
                        
                        # Try to find a column that might contain month names
                        for col in df.columns:
                            # Check for common month name patterns
                            if df[col].dtype == 'object':  # Check if column has string data
                                col_values = df[col].astype(str).str.lower().tolist()
                                # Check if any values match month names
                                month_patterns = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                                if any(any(pattern in val.lower() for pattern in month_patterns) for val in col_values if isinstance(val, str)):
                                    date_column = col
                                    has_month_names = True
                                    break
                        
                        # If month names detected, offer option to set custom year
                        if has_month_names:
                            st.info(f"Column '{date_column}' appears to contain month names without years.")
                            current_year = datetime.now().year
                            
                            with st.expander("Customize Date Settings", expanded=True):
                                st.markdown("### ⚠️ Month Names Detected - Set Custom Year")
                                st.write("The system detected month names without year information. By default, the system will use 2020 as the year. You can customize this below:")
                                
                                # Let user select the start year
                                custom_year = st.selectbox(
                                    "Select Start Year for This Data:",
                                    options=list(range(current_year - 10, current_year + 3)),
                                    index=10,  # Default to current year
                                    key="sop_custom_start_year"
                                )
                                
                                if st.button("Apply Custom Year", key="sop_apply_year_btn"):
                                    # Store original data before modification
                                    st.session_state['sop_original_data'] = df.copy()
                                    
                                    # Create a mapping function based on month detection
                                    def map_month_to_date(month_val, year=custom_year):
                                        month_val = str(month_val).lower().strip()
                                        month_map = {
                                            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                                            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                                            'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
                                            'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                                        }
                                        
                                        # Try to identify which month this is
                                        for key, value in month_map.items():
                                            if key in month_val:
                                                return pd.Timestamp(year=year, month=value, day=1)
                                        
                                        # If we couldn't match, return the original value
                                        return month_val
                                    
                                    try:
                                        # Create a copy to modify
                                        modified_df = df.copy()
                                        
                                        # Try to convert the date column
                                        modified_df[date_column] = modified_df[date_column].apply(map_month_to_date)
                                        
                                        # Update the dataframe
                                        df = modified_df
                                        st.session_state['sop_data'] = df
                                        
                                        # Show the modified data
                                        st.success(f"✅ Applied year {custom_year} to the data!")
                                        st.write("Updated data preview:")
                                        st.dataframe(df.head(10))
                                    except Exception as e:
                                        st.error(f"Error applying custom year: {e}")
                                        st.write("Please try again with different settings.")
                        
                        # Save to session state
                        st.session_state['sop_data'] = df
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            else:
                st.error(f"File not found: {file_path}")
                st.info("Please place your Excel file named 'sop_data.xlsx' in the app/data/sample_files directory")
                
        elif data_source == "Sample Data":
            st.info("Using sample S&OP data for demonstration.")
            st.session_state['sop_data'] = sample_df
            
        elif data_source == "Database Connection":
            st.info("Database connection feature will be available in the future.")
            st.session_state['sop_data'] = sample_df  # Use sample data for now
        
        # Display data preview
        if 'sop_data' in st.session_state:
            st.markdown("#### Data Preview")
            st.dataframe(st.session_state['sop_data'].head())
            
            # S&OP Dashboard setup
            st.markdown("#### S&OP Dashboard Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Planning horizon
                horizon = st.slider(
                    "Planning Horizon (months):",
                    min_value=3,
                    max_value=12,
                    value=6
                )
            
            with col2:
                # Target gross margin
                target_margin = st.slider(
                    "Target Gross Margin (%):",
                    min_value=20.0,
                    max_value=60.0,
                    value=40.0
                )
            
            # Store settings in session state
            st.session_state['sop_horizon'] = horizon
            st.session_state['sop_target_margin'] = target_margin
    
    # Demand-Supply Balance Tab
    with tab2:
        st.markdown("### Demand-Supply Balance")
        
        if 'sop_data' not in st.session_state:
            st.info("Please load S&OP data in the S&OP Data tab first.")
        else:
            st.markdown("""
            This tab shows the balance between demand and supply across your planning horizon.
            """)
            
            # Load data
            df = st.session_state['sop_data'].copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by product
            product_filter = st.selectbox(
                "Select Product:",
                ["All Products"] + df['product_name'].unique().tolist()
            )
            
            if product_filter != "All Products":
                df = df[df['product_name'] == product_filter]
            
            # Get horizon
            horizon = st.session_state.get('sop_horizon', 6)
            
            # Filter by date (last available date minus horizon)
            max_date = df['date'].max()
            min_date = max_date - pd.DateOffset(months=horizon-1)
            
            df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
            
            # Aggregate by date if All Products
            if product_filter == "All Products":
                df_agg = df.groupby('date').agg({
                    'forecast_demand': 'sum',
                    'planned_supply': 'sum',
                    'ending_inventory': 'sum'
                }).reset_index()
            else:
                df_agg = df
            
            # Plot demand vs supply
            fig = go.Figure()
            
            # Add demand
            fig.add_trace(go.Bar(
                x=df_agg['date'],
                y=df_agg['forecast_demand'],
                name='Demand',
                marker_color='#1f77b4'
            ))
            
            # Add supply
            fig.add_trace(go.Bar(
                x=df_agg['date'],
                y=df_agg['planned_supply'],
                name='Supply',
                marker_color='#ff7f0e'
            ))
            
            # Add inventory line
            fig.add_trace(go.Scatter(
                x=df_agg['date'],
                y=df_agg['ending_inventory'],
                mode='lines+markers',
                name='Ending Inventory',
                line=dict(color='#2ca02c', width=3)
            ))
            
            # Update layout
            fig.update_layout(
                title='Demand vs Supply Balance',
                xaxis_title='Month',
                yaxis_title='Units',
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            total_demand = df_agg['forecast_demand'].sum()
            total_supply = df_agg['planned_supply'].sum()
            balance = total_supply - total_demand
            balance_pct = (balance / total_demand) * 100 if total_demand > 0 else 0
            
            st.markdown("#### Supply-Demand Balance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Demand", f"{total_demand:,.0f}")
            
            with col2:
                st.metric("Total Supply", f"{total_supply:,.0f}")
            
            with col3:
                st.metric("Balance", f"{balance:,.0f}")
            
            with col4:
                st.metric("Balance %", f"{balance_pct:.1f}%")
            
            # Show imbalance alerts
            if abs(balance_pct) > 10:
                if balance_pct > 10:
                    st.warning(f"Supply exceeds demand by {balance_pct:.1f}%. Consider reducing production or finding new demand channels.")
                else:
                    st.error(f"Demand exceeds supply by {abs(balance_pct):.1f}%. Consider increasing production or prioritizing customers.")
            else:
                st.success("Demand and supply are well balanced (within ±10%).")
    
    # Financial Impact Tab
    with tab3:
        st.markdown("### Financial Impact")
        
        if 'sop_data' not in st.session_state:
            st.info("Please load S&OP data in the S&OP Data tab first.")
        else:
            st.markdown("""
            This tab shows the financial impact of your supply and demand plan.
            """)
            
            # Load data
            df = st.session_state['sop_data'].copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Get horizon and target margin
            horizon = st.session_state.get('sop_horizon', 6)
            target_margin = st.session_state.get('sop_target_margin', 40.0)
            
            # Filter by date
            max_date = df['date'].max()
            min_date = max_date - pd.DateOffset(months=horizon-1)
            
            df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
            
            # Aggregate by date
            df_agg = df.groupby('date').agg({
                'revenue': 'sum',
                'cogs': 'sum',
                'gross_margin': 'sum',
                'inventory_value': 'sum'
            }).reset_index()
            
            # Calculate gross margin percentage
            df_agg['gross_margin_pct'] = (df_agg['gross_margin'] / df_agg['revenue']) * 100
            
            # Plot financial metrics
            fig = go.Figure()
            
            # Add revenue
            fig.add_trace(go.Bar(
                x=df_agg['date'],
                y=df_agg['revenue'],
                name='Revenue',
                marker_color='#1f77b4'
            ))
            
            # Add COGS
            fig.add_trace(go.Bar(
                x=df_agg['date'],
                y=df_agg['cogs'],
                name='COGS',
                marker_color='#ff7f0e'
            ))
            
            # Add gross margin percentage line
            fig.add_trace(go.Scatter(
                x=df_agg['date'],
                y=df_agg['gross_margin_pct'],
                mode='lines+markers',
                name='Gross Margin %',
                yaxis='y2',
                line=dict(color='#2ca02c', width=3)
            ))
            
            # Add target margin line
            fig.add_shape(
                type='line',
                x0=min_date,
                y0=target_margin,
                x1=max_date,
                y1=target_margin,
                yref='y2',
                line=dict(
                    color='red',
                    width=2,
                    dash='dash'
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Financial Performance',
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                yaxis2=dict(
                    title='Gross Margin %',
                    overlaying='y',
                    side='right',
                    range=[0, max(df_agg['gross_margin_pct']) * 1.2]
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display financial metrics
            total_revenue = df_agg['revenue'].sum()
            total_cogs = df_agg['cogs'].sum()
            total_margin = df_agg['gross_margin'].sum()
            avg_margin_pct = (total_margin / total_revenue) * 100 if total_revenue > 0 else 0
            avg_inventory = df_agg['inventory_value'].mean()
            
            st.markdown("#### Financial Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Revenue", f"${total_revenue:,.0f}")
            
            with col2:
                st.metric("Total COGS", f"${total_cogs:,.0f}")
            
            with col3:
                st.metric("Total Gross Margin", f"${total_margin:,.0f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Gross Margin %", f"{avg_margin_pct:.1f}%", 
                         f"{avg_margin_pct - target_margin:.1f}%")
            
            with col2:
                st.metric("Average Inventory Value", f"${avg_inventory:,.0f}")
            
            # Show margin alerts
            if avg_margin_pct < target_margin:
                st.warning(f"Gross margin is below target by {target_margin - avg_margin_pct:.1f}%. Consider price adjustments or cost reductions.")
            else:
                st.success(f"Gross margin is meeting or exceeding the target of {target_margin:.1f}%.")
    
    # Executive Dashboard Tab
    with tab4:
        st.markdown("### Executive S&OP Dashboard")
        
        if 'sop_data' not in st.session_state:
            st.info("Please load S&OP data in the S&OP Data tab first.")
        else:
            st.markdown("""
            This dashboard provides a high-level view of key metrics for executive review.
            """)
            
            # Load data
            df = st.session_state['sop_data'].copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Get horizon
            horizon = st.session_state.get('sop_horizon', 6)
            
            # Filter by date
            max_date = df['date'].max()
            min_date = max_date - pd.DateOffset(months=horizon-1)
            
            df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
            
            # Aggregate by category and date
            category_view = df.groupby(['category', 'date']).agg({
                'forecast_demand': 'sum',
                'planned_supply': 'sum',
                'revenue': 'sum',
                'gross_margin': 'sum',
                'ending_inventory': 'sum'
            }).reset_index()
            
            # Calculate derived metrics
            category_view['gross_margin_pct'] = (category_view['gross_margin'] / category_view['revenue']) * 100
            category_view['inventory_turnover'] = category_view['forecast_demand'] / category_view['ending_inventory'].replace(0, 1)
            
            # Show category trends
            st.markdown("#### Category Performance Trends")
            
            metric_to_view = st.selectbox(
                "Select Metric to View:",
                ["Revenue", "Gross Margin %", "Demand", "Supply", "Inventory Turnover"]
            )
            
            # Map selection to column
            metric_map = {
                "Revenue": "revenue",
                "Gross Margin %": "gross_margin_pct",
                "Demand": "forecast_demand",
                "Supply": "planned_supply",
                "Inventory Turnover": "inventory_turnover"
            }
            
            selected_metric = metric_map[metric_to_view]
            
            # Plot category trends
            fig = px.line(
                category_view,
                x='date',
                y=selected_metric,
                color='category',
                title=f'Category {metric_to_view} Trends',
                labels={
                    'date': 'Month',
                    selected_metric: metric_to_view,
                    'category': 'Category'
                }
            )
            
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title=metric_to_view,
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Executive KPI summary
            st.markdown("#### Executive KPI Summary")
            
            # Aggregate by category
            category_summary = df.groupby('category').agg({
                'forecast_demand': 'sum',
                'planned_supply': 'sum',
                'revenue': 'sum',
                'gross_margin': 'sum',
                'inventory_value': 'mean'
            }).reset_index()
            
            # Calculate derived metrics
            category_summary['gross_margin_pct'] = (category_summary['gross_margin'] / category_summary['revenue']) * 100
            category_summary['balance'] = category_summary['planned_supply'] - category_summary['forecast_demand']
            category_summary['balance_pct'] = (category_summary['balance'] / category_summary['forecast_demand']) * 100
            
            # Display KPIs
            st.dataframe(category_summary.rename(columns={
                'category': 'Category',
                'forecast_demand': 'Demand',
                'planned_supply': 'Supply',
                'revenue': 'Revenue',
                'gross_margin': 'Gross Margin',
                'gross_margin_pct': 'Margin %',
                'balance': 'Balance',
                'balance_pct': 'Balance %',
                'inventory_value': 'Avg Inventory'
            }))
            
            # Show executive recommendation
            st.markdown("#### Executive Recommendation")
            
            # Calculate overall metrics
            total_revenue = df['revenue'].sum()
            total_margin = df['gross_margin'].sum()
            overall_margin_pct = (total_margin / total_revenue) * 100 if total_revenue > 0 else 0
            
            total_demand = df['forecast_demand'].sum()
            total_supply = df['planned_supply'].sum()
            overall_balance_pct = ((total_supply - total_demand) / total_demand) * 100 if total_demand > 0 else 0
            
            # Generate recommendation
            recommendation = ""
            
            if overall_margin_pct < target_margin:
                recommendation += f"- Margin is below target ({overall_margin_pct:.1f}% vs {target_margin:.1f}%): Consider price increases or cost reduction initiatives.\n\n"
            else:
                recommendation += f"- Margin is on or above target ({overall_margin_pct:.1f}% vs {target_margin:.1f}%): Focus on maintaining current pricing strategy and cost controls.\n\n"
            
            if abs(overall_balance_pct) > 10:
                if overall_balance_pct > 10:
                    recommendation += f"- Supply exceeds demand by {overall_balance_pct:.1f}%: Consider reducing production to avoid excess inventory costs.\n\n"
                else:
                    recommendation += f"- Demand exceeds supply by {abs(overall_balance_pct):.1f}%: Increase production capacity or implement customer prioritization.\n\n"
            else:
                recommendation += f"- Supply and demand are well balanced (within ±10%): Maintain current production levels.\n\n"
            
            # Find category with highest margin
            best_category = category_summary.loc[category_summary['gross_margin_pct'].idxmax()]
            recommendation += f"- Category {best_category['category']} shows the highest margin ({best_category['gross_margin_pct']:.1f}%): Consider shifting resources to this category.\n\n"
            
            # Executive summary
            st.info(recommendation)
            
            # Allow exporting to PDF
            st.markdown("#### Export Executive Dashboard")
            if st.button("Export to PDF"):
                st.success("Executive Dashboard exported to PDF (feature coming soon).")
