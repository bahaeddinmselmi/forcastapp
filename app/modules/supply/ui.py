"""
Supply Planning module UI components.
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

# Ensure app directory is in path
app_path = Path(__file__).parent.parent.parent
sys.path.append(str(app_path))

# Import config
import config

def generate_sample_supply_data():
    """Generate sample supply planning data for demonstration."""
    # Create sample product data
    products = [
        {"product_id": "P001", "name": "Premium Widget", "category": "A"},
        {"product_id": "P002", "name": "Standard Widget", "category": "B"},
        {"product_id": "P003", "name": "Economy Widget", "category": "C"},
        {"product_id": "P004", "name": "Deluxe Gadget", "category": "A"},
        {"product_id": "P005", "name": "Basic Gadget", "category": "B"},
    ]
    
    # Create resources (machines/production lines)
    resources = [
        {"resource_id": "R001", "name": "Production Line 1", "capacity_per_day": 100},
        {"resource_id": "R002", "name": "Production Line 2", "capacity_per_day": 150},
        {"resource_id": "R003", "name": "Assembly Station", "capacity_per_day": 200},
    ]
    
    # Create resource requirements for each product
    resource_requirements = []
    
    for product in products:
        for resource in resources:
            if np.random.random() > 0.3:  # 70% chance product needs this resource
                resource_requirements.append({
                    "product_id": product["product_id"],
                    "resource_id": resource["resource_id"],
                    "time_required": np.random.uniform(0.2, 2.0)  # hours per unit
                })
    
    # Create demand forecast for next 12 months
    start_date = datetime.now().replace(day=1)
    dates = [start_date + timedelta(days=30*i) for i in range(12)]
    
    # Create demand data
    demand_data = []
    
    for product in products:
        base_demand = np.random.randint(100, 500)
        
        for i, date in enumerate(dates):
            # Add some seasonality and trend
            seasonal_factor = 1.0 + 0.2 * np.sin(np.pi * i / 6)
            trend_factor = 1.0 + 0.03 * i
            
            # Calculate demand for this month
            monthly_demand = int(base_demand * seasonal_factor * trend_factor)
            
            demand_data.append({
                "product_id": product["product_id"],
                "product_name": product["name"],
                "date": date.strftime("%Y-%m-%d"),
                "forecast_demand": monthly_demand
            })
    
    # Create capacity constraints
    capacity_data = []
    
    for resource in resources:
        for i, date in enumerate(dates):
            # Base capacity with some variability
            capacity = resource["capacity_per_day"] * 22  # Assuming 22 working days
            
            # Add some variability (maintenance, holidays, etc.)
            capacity_factor = np.random.uniform(0.8, 1.0)
            monthly_capacity = int(capacity * capacity_factor)
            
            capacity_data.append({
                "resource_id": resource["resource_id"],
                "resource_name": resource["name"],
                "date": date.strftime("%Y-%m-%d"),
                "available_capacity": monthly_capacity
            })
    
    # Create raw material data
    materials = [
        {"material_id": "M001", "name": "Aluminum", "unit": "kg"},
        {"material_id": "M002", "name": "Plastic", "unit": "kg"},
        {"material_id": "M003", "name": "Steel", "unit": "kg"},
        {"material_id": "M004", "name": "Electronics", "unit": "pcs"},
    ]
    
    # Material requirements for each product
    bom_data = []
    
    for product in products:
        for material in materials:
            if np.random.random() > 0.3:  # 70% chance product needs this material
                bom_data.append({
                    "product_id": product["product_id"],
                    "material_id": material["material_id"],
                    "material_name": material["name"],
                    "quantity_per_unit": np.random.uniform(0.5, 5.0)
                })
    
    # Create dataframes
    products_df = pd.DataFrame(products)
    resources_df = pd.DataFrame(resources)
    resource_requirements_df = pd.DataFrame(resource_requirements)
    demand_df = pd.DataFrame(demand_data)
    capacity_df = pd.DataFrame(capacity_data)
    materials_df = pd.DataFrame(materials)
    bom_df = pd.DataFrame(bom_data)
    
    return {
        "products": products_df,
        "resources": resources_df,
        "resource_requirements": resource_requirements_df,
        "demand": demand_df,
        "capacity": capacity_df,
        "materials": materials_df,
        "bom": bom_df
    }

def show_supply_planning():
    """
    Display the Supply Planning module interface.
    """
    st.markdown("## Supply Planning")
    
    st.markdown("""
    The IBP for Supply module helps you manage production and supply planning 
    based on demand forecasts and capacity constraints.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Supply Data", 
        "Production Planning", 
        "Material Requirements",
        "Capacity Analysis"
    ])
    
    # Create sample data directory if it doesn't exist
    data_dir = os.path.join(app_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate sample data if it doesn't exist
    sample_data_path = os.path.join(data_dir, "sample_supply_data.json")
    if not os.path.exists(sample_data_path):
        sample_data = generate_sample_supply_data()
        
        # Save each dataframe as CSV
        for name, df in sample_data.items():
            df.to_csv(os.path.join(data_dir, f"sample_supply_{name}.csv"), index=False)
        
        # Also save a reference to the filenames
        with open(sample_data_path, 'w') as f:
            json.dump({name: f"sample_supply_{name}.csv" for name in sample_data.keys()}, f)
    
    # Load sample data references
    with open(sample_data_path, 'r') as f:
        data_files = json.load(f)
    
    # Function to load a specific dataset
    def load_dataset(name):
        return pd.read_csv(os.path.join(data_dir, data_files[name]))
    
    # Supply Data Tab
    with tab1:
        st.markdown("### Supply Data")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source:",
            ["Upload CSV", "Sample Data", "Database Connection"],
            horizontal=True
        )
        
        if data_source == "Upload CSV":
            st.info("CSV upload for multiple supply planning tables will be available in future versions.")
            st.session_state['use_sample_data'] = True
            
        elif data_source == "Sample Data":
            st.info("Using sample supply planning data for demonstration.")
            st.session_state['use_sample_data'] = True
            
        elif data_source == "Database Connection":
            st.info("Database connection feature will be available in the future.")
            st.session_state['use_sample_data'] = True
        
        # Load data
        if st.session_state.get('use_sample_data', False):
            # Load all datasets
            products_df = load_dataset('products')
            resources_df = load_dataset('resources')
            resource_requirements_df = load_dataset('resource_requirements')
            demand_df = load_dataset('demand')
            capacity_df = load_dataset('capacity')
            materials_df = load_dataset('materials')
            bom_df = load_dataset('bom')
            
            # Store in session state
            st.session_state['products_df'] = products_df
            st.session_state['resources_df'] = resources_df
            st.session_state['resource_requirements_df'] = resource_requirements_df
            st.session_state['demand_df'] = demand_df
            st.session_state['capacity_df'] = capacity_df
            st.session_state['materials_df'] = materials_df
            st.session_state['bom_df'] = bom_df
            
            # Display data previews in expandable sections
            with st.expander("Products"):
                st.dataframe(products_df)
            
            with st.expander("Resources (Production Lines)"):
                st.dataframe(resources_df)
            
            with st.expander("Resource Requirements"):
                st.dataframe(resource_requirements_df)
            
            with st.expander("Demand Forecast"):
                st.dataframe(demand_df)
            
            with st.expander("Capacity"):
                st.dataframe(capacity_df)
            
            with st.expander("Materials"):
                st.dataframe(materials_df)
            
            with st.expander("Bill of Materials"):
                st.dataframe(bom_df)
    
    # Production Planning Tab
    with tab2:
        st.markdown("### Production Planning")
        
        if not st.session_state.get('use_sample_data', False):
            st.info("Please load supply data in the Supply Data tab first.")
        else:
            st.markdown("""
            This tab helps you create a production plan based on demand forecasts and capacity constraints.
            """)
            
            # Load data from session state
            demand_df = st.session_state['demand_df']
            capacity_df = st.session_state['capacity_df']
            resource_requirements_df = st.session_state['resource_requirements_df']
            
            # Get unique products and dates
            products = demand_df['product_name'].unique()
            dates = pd.to_datetime(demand_df['date'].unique())
            
            # Filter options
            st.markdown("#### Planning Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Date range selection
                start_date = st.selectbox(
                    "Start Date:",
                    dates
                )
            
            with col2:
                # Planning horizon
                horizon = st.slider(
                    "Planning Horizon (months):",
                    min_value=1,
                    max_value=12,
                    value=6
                )
            
            # Filter demand data based on date range
            start_date = pd.to_datetime(start_date)
            end_date = start_date + pd.DateOffset(months=horizon)
            
            filtered_demand = demand_df[
                (pd.to_datetime(demand_df['date']) >= start_date) & 
                (pd.to_datetime(demand_df['date']) < end_date)
            ]
            
            # Group by product and sum demand
            product_demand = filtered_demand.groupby('product_name')['forecast_demand'].sum().reset_index()
            
            # Display product demand
            st.markdown("#### Product Demand (Planning Horizon)")
            st.dataframe(product_demand)
            
            # Production planning options
            st.markdown("#### Production Strategy")
            
            planning_strategy = st.selectbox(
                "Planning Strategy:",
                ["Level Production", "Chase Demand", "Hybrid Strategy"]
            )
            
            # Create production plan
            if st.button("Generate Production Plan"):
                with st.spinner("Generating production plan..."):
                    # Load the full demand data
                    full_demand = demand_df.copy()
                    full_demand['date'] = pd.to_datetime(full_demand['date'])
                    
                    # Filter for planning horizon
                    plan_demand = full_demand[
                        (full_demand['date'] >= start_date) & 
                        (full_demand['date'] < end_date)
                    ]
                    
                    # Create a production plan based on strategy
                    plan_data = []
                    
                    for product in plan_demand['product_name'].unique():
                        product_demand = plan_demand[plan_demand['product_name'] == product]
                        
                        for _, row in product_demand.iterrows():
                            if planning_strategy == "Level Production":
                                # Spread production evenly throughout the month
                                production_days = 22  # Assuming 22 working days per month
                                daily_production = row['forecast_demand'] / production_days
                                
                                plan_data.append({
                                    'product_name': product,
                                    'date': row['date'],
                                    'demand': row['forecast_demand'],
                                    'planned_production': row['forecast_demand'],
                                    'daily_production': daily_production,
                                    'strategy': 'Level'
                                })
                                
                            elif planning_strategy == "Chase Demand":
                                # Production follows demand exactly
                                plan_data.append({
                                    'product_name': product,
                                    'date': row['date'],
                                    'demand': row['forecast_demand'],
                                    'planned_production': row['forecast_demand'],
                                    'daily_production': row['forecast_demand'] / 22,
                                    'strategy': 'Chase'
                                })
                                
                            else:  # Hybrid
                                # Some smoothing of production
                                smoothing_factor = 0.3
                                
                                # Get previous month production if available
                                prev_month = row['date'] - pd.DateOffset(months=1)
                                prev_production = plan_demand[
                                    (plan_demand['product_name'] == product) & 
                                    (plan_demand['date'] == prev_month)
                                ]['forecast_demand'].sum()
                                
                                if prev_production == 0:
                                    planned_production = row['forecast_demand']
                                else:
                                    planned_production = (
                                        smoothing_factor * prev_production + 
                                        (1 - smoothing_factor) * row['forecast_demand']
                                    )
                                
                                plan_data.append({
                                    'product_name': product,
                                    'date': row['date'],
                                    'demand': row['forecast_demand'],
                                    'planned_production': planned_production,
                                    'daily_production': planned_production / 22,
                                    'strategy': 'Hybrid'
                                })
                    
                    # Create production plan dataframe
                    production_plan = pd.DataFrame(plan_data)
                    
                    # Store in session state
                    st.session_state['production_plan'] = production_plan
                    
                    # Display production plan
                    st.markdown("#### Production Plan")
                    st.dataframe(production_plan)
                    
                    # Plot production vs demand
                    fig = go.Figure()
                    
                    for product in production_plan['product_name'].unique():
                        product_data = production_plan[production_plan['product_name'] == product]
                        
                        fig.add_trace(go.Bar(
                            x=product_data['date'],
                            y=product_data['demand'],
                            name=f'{product} - Demand',
                            opacity=0.7
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=product_data['date'],
                            y=product_data['planned_production'],
                            mode='lines+markers',
                            name=f'{product} - Production',
                            line=dict(width=3)
                        ))
                    
                    fig.update_layout(
                        title=f'Production Plan vs Demand ({planning_strategy})',
                        xaxis_title='Month',
                        yaxis_title='Units',
                        legend=dict(x=0.01, y=0.99),
                        hovermode='x unified',
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Material Requirements Tab
    with tab3:
        st.markdown("### Material Requirements Planning")
        
        if not st.session_state.get('use_sample_data', False):
            st.info("Please load supply data in the Supply Data tab first.")
        elif 'production_plan' not in st.session_state:
            st.info("Please generate a production plan first.")
        else:
            st.markdown("""
            This tab calculates the materials needed to support your production plan.
            """)
            
            # Load data
            production_plan = st.session_state['production_plan']
            bom_df = st.session_state['bom_df']
            
            # Calculate material requirements
            if st.button("Calculate Material Requirements"):
                with st.spinner("Calculating material requirements..."):
                    # Merge production plan with product IDs
                    prod_with_id = pd.merge(
                        production_plan,
                        st.session_state['demand_df'][['product_name', 'product_id']].drop_duplicates(),
                        on='product_name'
                    )
                    
                    # Merge with BOM data
                    material_req = pd.merge(
                        prod_with_id,
                        bom_df,
                        on='product_id'
                    )
                    
                    # Calculate material quantities needed
                    material_req['material_quantity'] = material_req['planned_production'] * material_req['quantity_per_unit']
                    
                    # Group by material and date
                    material_by_date = material_req.groupby(['material_name', 'date'])['material_quantity'].sum().reset_index()
                    
                    # Store in session state
                    st.session_state['material_requirements'] = material_by_date
                    
                    # Display material requirements
                    st.markdown("#### Material Requirements by Month")
                    
                    # Pivot table to show materials by month
                    pivot_table = material_by_date.pivot_table(
                        index='material_name',
                        columns='date',
                        values='material_quantity',
                        aggfunc='sum'
                    ).fillna(0)
                    
                    st.dataframe(pivot_table)
                    
                    # Plot material requirements
                    fig = px.line(
                        material_by_date,
                        x='date',
                        y='material_quantity',
                        color='material_name',
                        title='Material Requirements Over Time',
                        labels={
                            'date': 'Month',
                            'material_quantity': 'Quantity Required',
                            'material_name': 'Material'
                        }
                    )
                    
                    fig.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Quantity Required',
                        legend=dict(x=0.01, y=0.99),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Total material requirements
                    total_materials = material_by_date.groupby('material_name')['material_quantity'].sum().reset_index()
                    
                    # Display total material requirements
                    st.markdown("#### Total Material Requirements (Planning Horizon)")
                    
                    fig = px.bar(
                        total_materials,
                        x='material_name',
                        y='material_quantity',
                        title='Total Material Requirements',
                        labels={
                            'material_name': 'Material',
                            'material_quantity': 'Total Quantity Required'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Capacity Analysis Tab
    with tab4:
        st.markdown("### Capacity Analysis")
        
        if not st.session_state.get('use_sample_data', False):
            st.info("Please load supply data in the Supply Data tab first.")
        elif 'production_plan' not in st.session_state:
            st.info("Please generate a production plan first.")
        else:
            st.markdown("""
            This tab analyzes whether your production plan is feasible given resource constraints.
            """)
            
            # Load data
            production_plan = st.session_state['production_plan']
            resource_requirements_df = st.session_state['resource_requirements_df']
            capacity_df = st.session_state['capacity_df']
            
            # Capacity analysis
            if st.button("Analyze Capacity Requirements"):
                with st.spinner("Analyzing capacity requirements..."):
                    # Merge production plan with product IDs
                    prod_with_id = pd.merge(
                        production_plan,
                        st.session_state['demand_df'][['product_name', 'product_id']].drop_duplicates(),
                        on='product_name'
                    )
                    
                    # Merge with resource requirements
                    resource_needed = pd.merge(
                        prod_with_id,
                        resource_requirements_df,
                        on='product_id'
                    )
                    
                    # Calculate capacity required (hours)
                    resource_needed['capacity_required'] = resource_needed['planned_production'] * resource_needed['time_required']
                    
                    # Merge with resource names
                    resource_needed = pd.merge(
                        resource_needed,
                        st.session_state['resources_df'][['resource_id', 'name']],
                        left_on='resource_id',
                        right_on='resource_id'
                    )
                    
                    # Group by resource and date
                    capacity_by_date = resource_needed.groupby(['name', 'date'])['capacity_required'].sum().reset_index()
                    
                    # Get capacity data in same format
                    capacity_data = capacity_df.copy()
                    capacity_data['date'] = pd.to_datetime(capacity_data['date'])
                    capacity_data = capacity_data[['resource_name', 'date', 'available_capacity']]
                    
                    # Merge capacity required with capacity available
                    capacity_combined = pd.merge(
                        capacity_by_date,
                        capacity_data,
                        left_on=['name', 'date'],
                        right_on=['resource_name', 'date'],
                        how='left'
                    )
                    
                    # Calculate capacity utilization
                    capacity_combined['utilization'] = (capacity_combined['capacity_required'] / capacity_combined['available_capacity']) * 100
                    capacity_combined['status'] = np.where(capacity_combined['utilization'] <= 100, 'OK', 'Over Capacity')
                    
                    # Store in session state
                    st.session_state['capacity_analysis'] = capacity_combined
                    
                    # Display capacity analysis
                    st.markdown("#### Resource Utilization by Month")
                    
                    # Pivot table to show utilization by month
                    pivot_table = capacity_combined.pivot_table(
                        index='name',
                        columns='date',
                        values='utilization',
                        aggfunc='sum'
                    ).fillna(0)
                    
                    st.dataframe(pivot_table)
                    
                    # Plot capacity utilization
                    fig = px.line(
                        capacity_combined,
                        x='date',
                        y='utilization',
                        color='name',
                        title='Resource Utilization Over Time',
                        labels={
                            'date': 'Month',
                            'utilization': 'Utilization (%)',
                            'name': 'Resource'
                        }
                    )
                    
                    # Add a horizontal line at 100%
                    fig.add_shape(
                        type='line',
                        x0=capacity_combined['date'].min(),
                        y0=100,
                        x1=capacity_combined['date'].max(),
                        y1=100,
                        line=dict(
                            color='red',
                            width=2,
                            dash='dash'
                        )
                    )
                    
                    fig.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Utilization (%)',
                        legend=dict(x=0.01, y=0.99),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify bottlenecks
                    bottlenecks = capacity_combined[capacity_combined['status'] == 'Over Capacity']
                    
                    if not bottlenecks.empty:
                        st.markdown("#### Capacity Bottlenecks")
                        st.warning("The following resources are over capacity:")
                        st.dataframe(bottlenecks[['name', 'date', 'capacity_required', 'available_capacity', 'utilization']])
                        
                        # Suggestions for resolving bottlenecks
                        st.markdown("#### Bottleneck Resolution Options")
                        
                        st.markdown("""
                        Options to resolve capacity bottlenecks:
                        - Increase capacity (overtime, additional shifts)
                        - Outsource production
                        - Adjust production plan
                        - Prioritize products based on profitability
                        """)
                    else:
                        st.success("No capacity bottlenecks detected. The production plan is feasible.")
                    
                    # Average utilization by resource
                    avg_utilization = capacity_combined.groupby('name')['utilization'].mean().reset_index()
                    
                    # Display average utilization
                    st.markdown("#### Average Resource Utilization")
                    
                    fig = px.bar(
                        avg_utilization,
                        x='name',
                        y='utilization',
                        title='Average Resource Utilization',
                        labels={
                            'name': 'Resource',
                            'utilization': 'Average Utilization (%)'
                        }
                    )
                    
                    # Add a reference line for target utilization
                    fig.add_shape(
                        type='line',
                        x0=-0.5,
                        y0=85,
                        x1=len(avg_utilization) - 0.5,
                        y1=85,
                        line=dict(
                            color='green',
                            width=2,
                            dash='dash'
                        )
                    )
                    
                    fig.add_annotation(
                        x=0,
                        y=85,
                        text="Target Utilization",
                        showarrow=False,
                        yshift=10
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
