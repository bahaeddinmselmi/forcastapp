"""
Inventory Optimization module UI components with fixed widget keys.
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

# Get currency settings
CURRENCY_SETTINGS = config.CURRENCY_SETTINGS
DEFAULT_CURRENCY = config.DEFAULT_CURRENCY

# Import export utilities
from utils.export import create_excel_download_link, create_full_report_download_link

# Import Excel detector
from utils.excel_detector import load_excel_with_smart_detection

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

def safe_get(data, key, default=0):
    """Safely get a value from a dictionary or Series."""
    try:
        return data[key] if key in data else default
    except:
        return default
        
def format_currency(value, currency_code):
    """Format a value with the appropriate currency symbol."""
    if currency_code in CURRENCY_SETTINGS:
        symbol = CURRENCY_SETTINGS[currency_code]['symbol']
        
        # Special formatting for EUR (Euro)
        if currency_code == "EUR":
            # European format: symbol after the value with space, comma as decimal separator
            return f"{value:,.2f} {symbol}"
        # Special formatting for TND
        elif currency_code == "TND":
            # Middle Eastern format: symbol first
            return f"{symbol} {value:,.3f}"
        else:
            # Default format for other currencies
            return f"{symbol}{value:,.2f}"
    else:
        return f"${value:,.2f}"  # Default to USD if currency code not found

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
    
    # Currency selection in sidebar
    with st.sidebar:
        st.markdown("### Currency Settings")
        currency_options = [f"{code} - {details['name']} ({details['symbol']})" for code, details in CURRENCY_SETTINGS.items()]
        selected_currency_full = st.selectbox(
            "Select Currency:", 
            options=currency_options,
            index=list(CURRENCY_SETTINGS.keys()).index(DEFAULT_CURRENCY) if DEFAULT_CURRENCY in CURRENCY_SETTINGS else 0,
            key="currency_selector"
        )
        
        # Extract currency code from selection
        selected_currency = selected_currency_full.split(" - ")[0] if selected_currency_full else DEFAULT_CURRENCY
        
        # Store in session state
        st.session_state['selected_currency'] = selected_currency
        st.session_state['currency_symbol'] = CURRENCY_SETTINGS[selected_currency]['symbol']
        
        # Show currently selected currency
        st.success(f"Using {CURRENCY_SETTINGS[selected_currency]['name']} ({CURRENCY_SETTINGS[selected_currency]['symbol']})")
        
        # Add a small currency converter if needed
        with st.expander("Need to convert values?"):
            base_amount = st.number_input("Amount to convert:", min_value=0.0, value=100.0, step=10.0, key="convert_amount")
            st.write(f"**{base_amount} {selected_currency}** is approximately:")
            
            # Show conversions to a few major currencies
            sample_rates = {
                'USD': 0.32, 'EUR': 0.29, 'GBP': 0.25, 'TND': 1.0, 'JPY': 49.26, 'CNY': 2.32
            }
            
            # Get approximate conversions
            if selected_currency == 'TND':
                for curr, rate in sample_rates.items():
                    if curr != selected_currency:
                        st.write(f"- {format_currency(base_amount * rate, curr)} ({curr})")
            else:
                # Convert from other currency to TND (just for example)
                tnd_rate = 1.0 / sample_rates[selected_currency] if selected_currency in sample_rates else 3.1
                st.write(f"- {format_currency(base_amount * tnd_rate, 'TND')} (TND)")
                
            st.caption("Note: These are approximate values for demonstration purposes.")
            
    # Create horizontal line for separation
    st.markdown("---")

    
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
            ["Upload Data", "Load From Directory", "Sample Data", "Database Connection"],
            horizontal=True,
            key="data_source_radio"
        )
        
        if data_source == "Upload Data":
            uploaded_file = st.file_uploader("Upload your inventory data:", type=["csv", "xlsx", "xls"], key="inventory_uploader",
                help="Support for CSV (.csv) and Excel (.xlsx, .xls) files")
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
                                key="inventory_manual_sheet"
                            )
                            
                            # Options for data range
                            col1, col2 = st.columns(2)
                            with col1:
                                start_row = st.number_input("Start Row (0-based)", min_value=0, value=0, key="inventory_start_row")
                                has_header = st.checkbox("First row is header", value=True, key="inventory_manual_header")
                            
                            with col2:
                                end_row = st.number_input("End Row (leave at 1000 to read all)", min_value=1, value=1000, key="inventory_end_row")
                                first_col = st.text_input("First Column (e.g., A)", value="A", key="inventory_first_col")
                                last_col = st.text_input("Last Column (e.g., Z or leave empty for all)", value="", key="inventory_last_col")
                            
                            # Convert Excel column letters to usecols parameter
                            def get_usecols_param(first, last):
                                if not first:
                                    return None
                                if not last:
                                    return first + ":" if first else None
                                return f"{first}:{last}"
                            
                            usecols = get_usecols_param(first_col, last_col)
                            
                            # Preview button with manual settings
                            if st.button("Preview with Manual Settings", key="inventory_manual_preview"):
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
                                    if st.button("Use These Settings", key="inventory_use_manual"):
                                        df = preview_df
                                        st.session_state['inventory_override_active'] = True
                                        st.success("Using manual settings for data import!")
                                except Exception as e:
                                    st.error(f"Error with manual settings: {e}")
                        
                        # Only use AI detection if manual override is not active
                        if not st.session_state.get('inventory_override_active', False):
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
                                if len(df) > 15 and st.button("Show more rows", key="inventory_more_rows"):
                                    st.dataframe(df)
                    else:  # Default to CSV
                        df = pd.read_csv(uploaded_file)
                        
                    # Save data to session state
                    st.session_state['inventory_data'] = df
                    st.success(f"Successfully loaded {file_type.upper()} data with {df.shape[0]} rows and {df.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            else:
                st.info("Please upload your inventory data file (Excel or CSV).")
                st.caption("Our AI will automatically detect data tables in your file, even if they're not at the top of the sheet.")
                
        elif data_source == "Load From Directory":
            st.info("Loading data from the app/data/sample_files directory")
            file_path = os.path.join(app_path, "data", "sample_files", "inventory_data.xlsx")
            
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
                            if len(df) > 15 and st.button("Show more rows", key="inv_dir_more_rows"):
                                st.dataframe(df)
                        
                        # Save to session state
                        st.session_state['inventory_data'] = df
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            else:
                st.error(f"File not found: {file_path}")
                st.info("Please place your Excel file named 'inventory_data.xlsx' in the app/data/sample_files directory")
                
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
            
            # Show column information
            df = st.session_state['inventory_data']
            columns_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Values": df.count().values,
                "Null Values": df.isna().sum().values
            })
            
            with st.expander("Dataset Column Information"):
                st.dataframe(columns_info)
                
                # Check for expected columns
                expected_columns = ['annual_demand', 'unit_cost', 'lead_time_days', 'order_cost', 
                                    'holding_cost_pct', 'current_stock', 'std_dev_demand']
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"Missing expected columns: {', '.join(missing_columns)}. Some features may not work properly.")
                    st.info("Required columns for full functionality: product_id, name, annual_demand, unit_cost, lead_time_days, order_cost, holding_cost_pct, current_stock, std_dev_demand")
            
            # ABC Analysis
            st.markdown("#### ABC Analysis")
            
            if st.button("Run ABC Analysis", key="run_abc_button"):
                # Get data
                df = st.session_state['inventory_data'].copy()
                
                # Check for required columns
                if 'annual_demand' not in df.columns or 'unit_cost' not in df.columns:
                    st.error("Cannot perform ABC Analysis: Missing required columns 'annual_demand' or 'unit_cost'")
                else:
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
                    
                    # Prepare columns for display
                    display_cols = ['class', 'annual_value', 'percentage', 'cumulative_percentage']
                    product_cols = []
                    
                    for col in ['product_id', 'name', 'sku', 'product_name', 'item_name', 'description']:
                        if col in df.columns:
                            product_cols.append(col)
                    
                    display_cols = product_cols + display_cols
                    
                    # Display results
                    st.dataframe(df[display_cols])
                    
                    # Create Pareto chart
                    fig = go.Figure()
                    
                    # Use appropriate column for x-axis labels
                    x_col = product_cols[0] if product_cols else df.index
                    
                    # Add annual value bars
                    fig.add_trace(go.Bar(
                        x=df[x_col] if isinstance(x_col, str) else x_col,
                        y=df['annual_value'],
                        name='Annual Value',
                        marker_color='blue'
                    ))
                    
                    # Add cumulative percentage line
                    fig.add_trace(go.Scatter(
                        x=df[x_col] if isinstance(x_col, str) else x_col,
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
            # Get inventory data
            df = st.session_state['inventory_data']
            
            # Check if we have the required columns
            required_columns = ['annual_demand', 'unit_cost', 'order_cost', 'holding_cost_pct']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Missing required columns for EOQ Analysis: {', '.join(missing_columns)}")
                st.info("This analysis requires: annual_demand, unit_cost, order_cost, holding_cost_pct.")
                st.info("Sample data will be used for missing columns.")
            
            # Product selection
            st.markdown("#### Select Product")
            
            # Product selection - handle different column name possibilities
            product_name_column = None
            
            # Try to identify product name column
            possible_name_columns = ['name', 'product_name', 'item_name', 'description', 'product_id', 'sku']
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
                df[product_name_column].tolist(),
                key="eoq_product_selectbox"  # Add unique key
            )
            
            # Get selected product data
            product_data = df[df[product_name_column] == selected_product].iloc[0]
            
            # Display product details
            st.markdown("#### Product Details")
            
            col1, col2, col3 = st.columns(3)
            
            # Get currency code from session state
            currency_code = st.session_state.get('selected_currency', DEFAULT_CURRENCY)
            
            with col1:
                annual_demand = safe_get(product_data, 'annual_demand', 5000)
                st.metric("Annual Demand", f"{annual_demand:,.0f} units")
                
            with col2:
                unit_cost = safe_get(product_data, 'unit_cost', 25.0)
                st.metric("Unit Cost", format_currency(unit_cost, currency_code))
                
            with col3:
                lead_time_days = safe_get(product_data, 'lead_time_days', 30)
                st.metric("Lead Time", f"{lead_time_days} days")
            
            # EOQ Parameters
            st.markdown("#### EOQ Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                annual_demand_input = st.number_input(
                    "Annual Demand (units):",
                    min_value=1.0,
                    value=float(annual_demand),
                    step=100.0,
                    key="eoq_annual_demand_input"  # Add unique key
                )
                
                # Get currency symbol for display
                currency_symbol = st.session_state.get('currency_symbol', '$')
                
                order_cost = st.number_input(
                    f"Order Cost ({currency_symbol}):",
                    min_value=1.0,
                    value=float(safe_get(product_data, 'order_cost', 100.0)),
                    step=10.0,
                    key="eoq_order_cost_input"  # Add unique key
                )
                
            with col2:
                holding_cost_pct = st.slider(
                    "Holding Cost (% of unit cost):",
                    min_value=1.0,
                    max_value=50.0,
                    value=float(safe_get(product_data, 'holding_cost_pct', 0.25) * 100),
                    step=1.0,
                    key="eoq_holding_cost_slider"  # Add unique key
                ) / 100
                
                unit_cost_input = st.number_input(
                    f"Unit Cost ({currency_symbol}):",
                    min_value=0.01,
                    value=float(unit_cost),
                    step=1.0,
                    key="eoq_unit_cost_input"  # Add unique key
                )
            
            # Calculate EOQ
            holding_cost = unit_cost_input * holding_cost_pct
            eoq = calculate_eoq(annual_demand_input, order_cost, holding_cost)
            annual_orders = annual_demand_input / eoq
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
                
            # Calculate annual costs
            annual_order_cost = annual_orders * order_cost
            annual_holding_cost = (eoq / 2) * holding_cost
            total_annual_cost = annual_order_cost + annual_holding_cost
            
            # Display annual cost analysis
            st.markdown("#### Annual Cost Analysis")
            cost_col1, cost_col2, cost_col3 = st.columns(3)
            
            with cost_col1:
                st.metric("Annual Order Cost", format_currency(annual_order_cost, currency_code))
                
            with cost_col2:
                st.metric("Annual Holding Cost", format_currency(annual_holding_cost, currency_code))
                
            with cost_col3:
                st.metric("Total Annual Cost", format_currency(total_annual_cost, currency_code))
            
            # Create a dataframe for export
            eoq_data = pd.DataFrame([
                {"Parameter": "Product", "Value": product_name},
                {"Parameter": "Economic Order Quantity (EOQ)", "Value": f"{eoq:.0f} units"},
                {"Parameter": "Annual Demand", "Value": f"{annual_demand_input:.0f} units"},
                {"Parameter": "Order Cost", "Value": format_currency(order_cost, currency_code)},
                {"Parameter": "Unit Cost", "Value": format_currency(unit_cost_input, currency_code)},
                {"Parameter": "Holding Cost Rate", "Value": f"{holding_cost_pct:.1%}"},
                {"Parameter": "Holding Cost per Unit", "Value": format_currency(holding_cost, currency_code)},
                {"Parameter": "Annual Order Cost", "Value": format_currency(annual_order_cost, currency_code)},
                {"Parameter": "Annual Holding Cost", "Value": format_currency(annual_holding_cost, currency_code)},
                {"Parameter": "Total Annual Cost", "Value": format_currency(total_annual_cost, currency_code)},
                {"Parameter": "Order Cycle", "Value": f"{order_cycle_days:.1f} days"},
            ])
            
            # Add export options
            st.markdown("#### Export Options")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st.markdown(create_excel_download_link(eoq_data, "EOQ Analysis", "EOQ Analysis"), unsafe_allow_html=True)
            with export_col2:
                # Create metadata for full report
                metadata = {
                    "Generated on": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Product": product_name,
                    "Currency": currency_code,
                    "Analysis Type": "Economic Order Quantity (EOQ)"
                }
                
                # Generate full report download link
                st.markdown(create_full_report_download_link(
                    eoq_data, 
                    "Inventory Optimization Report", 
                    None,  # We'll add chart support later
                    metadata
                ), unsafe_allow_html=True)
            
    # Replenishment Planning Tab
    with tab3:
        st.markdown("### Replenishment Planning")
        
        if 'inventory_data' not in st.session_state:
            st.info("Please load inventory data in the Inventory Data tab first.")
        else:
            # Get inventory data
            df = st.session_state['inventory_data']
            
            # Check if we have the required columns
            required_columns = ['daily_demand', 'std_dev_demand', 'current_stock', 'lead_time_days']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Missing required columns for Replenishment Planning: {', '.join(missing_columns)}")
                st.info("This analysis requires: daily_demand, std_dev_demand, current_stock, lead_time_days.")
                st.info("Sample data will be used for missing columns.")
            
            # Product selection - handle different column name possibilities
            product_name_column = None
            
            # Try to identify product name column
            possible_name_columns = ['name', 'product_name', 'item_name', 'description', 'product_id', 'sku']
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
                df[product_name_column].tolist(),
                key="replenishment_product_selectbox"  # Add unique key
            )
            
            # Get selected product data
            product_data = df[df[product_name_column] == selected_product].iloc[0]
            
            # Service level selection
            service_level = st.slider(
                "Service Level (%):",
                min_value=90,
                max_value=99,
                value=95,
                step=1,
                key="replenishment_service_level_slider"  # Add unique key
            ) / 100
            
            # Get values with defaults for missing columns
            daily_demand = safe_get(product_data, 'daily_demand', 
                                  safe_get(product_data, 'annual_demand', 5000) / 365)
            std_dev_demand = safe_get(product_data, 'std_dev_demand', daily_demand * 0.1)
            lead_time_days = safe_get(product_data, 'lead_time_days', 30)
            current_stock = safe_get(product_data, 'current_stock', 500)
            
            # Calculate safety stock
            safety_stock = calculate_safety_stock(
                service_level,
                lead_time_days,
                std_dev_demand
            )
            
            # Calculate reorder point
            reorder_point = calculate_reorder_point(
                lead_time_days,
                daily_demand,
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
                st.metric("Current Stock", f"{current_stock:.0f} units")
            
            # Stock status
            stock_status = "OK"
            days_to_stockout = 0
            
            if current_stock <= reorder_point:
                if current_stock <= safety_stock:
                    stock_status = "CRITICAL"
                else:
                    stock_status = "ORDER NOW"
                days_to_stockout = current_stock / daily_demand if daily_demand > 0 else float('inf')
            
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
                # Get currency code from session state
                currency_code = st.session_state.get('selected_currency', DEFAULT_CURRENCY)
                
                # Calculate EOQ
                unit_cost = safe_get(product_data, 'unit_cost', 25.0)
                holding_cost_pct = safe_get(product_data, 'holding_cost_pct', 0.25)
                holding_cost = unit_cost * holding_cost_pct
                order_cost = safe_get(product_data, 'order_cost', 100.0)
                annual_demand = safe_get(product_data, 'annual_demand', daily_demand * 365)
                
                eoq = calculate_eoq(
                    annual_demand,
                    order_cost,
                    holding_cost
                )
                
                st.markdown("#### Recommended Order")
                
                # Calculate order cost in current currency
                order_cost_total = eoq * unit_cost
                
                st.info(f"Place an order for {eoq:.0f} units as soon as possible.\n\nEstimated cost: {format_currency(order_cost_total, currency_code)}")
