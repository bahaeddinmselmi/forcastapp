"""
Control Tower module UI components.
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
import random

# Ensure app directory is in path
app_path = Path(__file__).parent.parent.parent
sys.path.append(str(app_path))

# Import config
import config

def generate_sample_kpi_data():
    """Generate sample KPI data for the control tower."""
    # Create date range for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(31)]
    
    # Create KPI data
    kpi_data = []
    
    # Base values for KPIs
    base_values = {
        "order_fill_rate": 95.0,
        "on_time_delivery": 92.0,
        "forecast_accuracy": 85.0,
        "inventory_turns": 8.0,
        "perfect_order": 88.0,
        "cash_to_cash": 45.0,
        "supply_chain_cost": 12.0
    }
    
    # Create time series for each KPI
    for date in dates:
        # Add some random variation and minor trend
        day_factor = (date - start_date).days / 30  # 0 to 1 over the period
        
        # Each KPI varies differently
        kpi_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "order_fill_rate": base_values["order_fill_rate"] - 1.5 * day_factor + random.uniform(-1, 1),
            "on_time_delivery": base_values["on_time_delivery"] - 2 * day_factor + random.uniform(-1.5, 1),
            "forecast_accuracy": base_values["forecast_accuracy"] + 2.5 * day_factor + random.uniform(-2, 2),
            "inventory_turns": base_values["inventory_turns"] + 0.8 * day_factor + random.uniform(-0.3, 0.3),
            "perfect_order": base_values["perfect_order"] - 1 * day_factor + random.uniform(-1.5, 1.5),
            "cash_to_cash": base_values["cash_to_cash"] - 3 * day_factor + random.uniform(-2, 2),
            "supply_chain_cost": base_values["supply_chain_cost"] + 0.5 * day_factor + random.uniform(-0.5, 0.5)
        })
    
    return pd.DataFrame(kpi_data)

def generate_sample_alerts():
    """Generate sample alerts for the control tower."""
    alert_types = [
        "Inventory Alert", "Delivery Alert", "Forecast Alert", "Quality Alert", 
        "Supplier Alert", "Demand Alert", "Capacity Alert", "Cost Alert"
    ]
    
    severity_levels = ["Critical", "High", "Medium", "Low"]
    
    alerts = []
    
    # Generate 15 sample alerts
    for i in range(15):
        alert_type = random.choice(alert_types)
        severity = random.choice(severity_levels)
        
        # Date within the last 7 days
        days_ago = random.randint(0, 7)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M")
        
        # Generate alert message based on type
        if alert_type == "Inventory Alert":
            item = random.choice(["Premium Widget", "Standard Widget", "Deluxe Gadget"])
            message = f"{item} inventory below safety stock level"
        elif alert_type == "Delivery Alert":
            message = f"Shipment delayed for order #{random.randint(10000, 99999)}"
        elif alert_type == "Forecast Alert":
            message = f"Forecast accuracy below threshold for product category"
        elif alert_type == "Quality Alert":
            message = f"Quality issue detected in batch #{random.randint(1000, 9999)}"
        elif alert_type == "Supplier Alert":
            message = f"Supplier performance issue detected"
        elif alert_type == "Demand Alert":
            message = f"Unexpected demand spike detected"
        elif alert_type == "Capacity Alert":
            message = f"Production line capacity at {random.randint(90, 99)}%"
        elif alert_type == "Cost Alert":
            message = f"Transportation costs exceeding budget by {random.randint(5, 20)}%"
        
        alerts.append({
            "date": date,
            "type": alert_type,
            "severity": severity,
            "message": message,
            "status": random.choice(["New", "Acknowledged", "In Progress", "Resolved"])
        })
    
    # Sort by date (most recent first) and severity
    return pd.DataFrame(alerts).sort_values(by=["date", "severity"], ascending=[False, True])

def show_control_tower():
    """
    Display the Control Tower module interface.
    """
    st.markdown("## Supply Chain Control Tower")
    
    st.markdown("""
    The IBP Control Tower provides a real-time, end-to-end view of supply chain performance,
    KPIs, and alerts to help you monitor and respond to changes in your supply chain.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "KPI Dashboard", 
        "Alert Management", 
        "Process Monitoring",
        "Supply Chain Visibility"
    ])
    
    # Create sample data directory if it doesn't exist
    data_dir = os.path.join(app_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate and save sample KPI data if it doesn't exist
    kpi_data_path = os.path.join(data_dir, "sample_kpi_data.csv")
    if not os.path.exists(kpi_data_path):
        kpi_df = generate_sample_kpi_data()
        kpi_df.to_csv(kpi_data_path, index=False)
    
    # Load KPI data
    kpi_df = pd.read_csv(kpi_data_path)
    kpi_df["date"] = pd.to_datetime(kpi_df["date"])
    
    # Generate and save sample alerts if they don't exist
    alerts_path = os.path.join(data_dir, "sample_alerts.csv")
    if not os.path.exists(alerts_path):
        alerts_df = generate_sample_alerts()
        alerts_df.to_csv(alerts_path, index=False)
    
    # Load alerts
    alerts_df = pd.read_csv(alerts_path)
    
    # KPI Dashboard Tab
    with tab1:
        st.markdown("### KPI Dashboard")
        
        # Get most recent KPI values
        latest_kpis = kpi_df.iloc[-1]
        previous_kpis = kpi_df.iloc[-8]  # Compare to week ago
        
        # Create metrics row
        st.markdown("#### Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current = latest_kpis["order_fill_rate"]
            previous = previous_kpis["order_fill_rate"]
            delta = current - previous
            st.metric(
                "Order Fill Rate", 
                f"{current:.1f}%", 
                f"{delta:.1f}%",
                delta_color="normal" if delta >= 0 else "inverse"
            )
        
        with col2:
            current = latest_kpis["on_time_delivery"]
            previous = previous_kpis["on_time_delivery"]
            delta = current - previous
            st.metric(
                "On-Time Delivery", 
                f"{current:.1f}%", 
                f"{delta:.1f}%",
                delta_color="normal" if delta >= 0 else "inverse"
            )
        
        with col3:
            current = latest_kpis["forecast_accuracy"]
            previous = previous_kpis["forecast_accuracy"]
            delta = current - previous
            st.metric(
                "Forecast Accuracy", 
                f"{current:.1f}%", 
                f"{delta:.1f}%",
                delta_color="normal" if delta >= 0 else "inverse"
            )
        
        with col4:
            current = latest_kpis["inventory_turns"]
            previous = previous_kpis["inventory_turns"]
            delta = current - previous
            st.metric(
                "Inventory Turns", 
                f"{current:.1f}", 
                f"{delta:.1f}",
                delta_color="normal" if delta >= 0 else "inverse"
            )
        
        # Second row of metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current = latest_kpis["perfect_order"]
            previous = previous_kpis["perfect_order"]
            delta = current - previous
            st.metric(
                "Perfect Order %", 
                f"{current:.1f}%", 
                f"{delta:.1f}%",
                delta_color="normal" if delta >= 0 else "inverse"
            )
        
        with col2:
            current = latest_kpis["cash_to_cash"]
            previous = previous_kpis["cash_to_cash"]
            delta = current - previous
            st.metric(
                "Cash-to-Cash Cycle", 
                f"{current:.1f} days", 
                f"{delta:.1f} days",
                delta_color="inverse" if delta >= 0 else "normal"
            )
        
        with col3:
            current = latest_kpis["supply_chain_cost"]
            previous = previous_kpis["supply_chain_cost"]
            delta = current - previous
            st.metric(
                "Supply Chain Cost", 
                f"{current:.1f}% of Revenue", 
                f"{delta:.1f}%",
                delta_color="inverse" if delta >= 0 else "normal"
            )
        
        # KPI trends
        st.markdown("#### KPI Trends")
        
        # Select KPIs to view
        selected_kpis = st.multiselect(
            "Select KPIs to View:",
            ["order_fill_rate", "on_time_delivery", "forecast_accuracy", "inventory_turns", "perfect_order", "cash_to_cash", "supply_chain_cost"],
            default=["order_fill_rate", "on_time_delivery", "forecast_accuracy"]
        )
        
        # Map KPI names to display names
        kpi_display_names = {
            "order_fill_rate": "Order Fill Rate",
            "on_time_delivery": "On-Time Delivery",
            "forecast_accuracy": "Forecast Accuracy",
            "inventory_turns": "Inventory Turns",
            "perfect_order": "Perfect Order %",
            "cash_to_cash": "Cash-to-Cash Cycle",
            "supply_chain_cost": "Supply Chain Cost"
        }
        
        # Plot selected KPIs
        if selected_kpis:
            fig = go.Figure()
            
            for kpi in selected_kpis:
                fig.add_trace(go.Scatter(
                    x=kpi_df["date"],
                    y=kpi_df[kpi],
                    mode="lines+markers",
                    name=kpi_display_names.get(kpi, kpi)
                ))
            
            fig.update_layout(
                title="KPI Trends (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Value",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # KPI heatmap
        st.markdown("#### KPI Performance Heatmap")
        
        # Get last 7 days of data
        last_7_days = kpi_df.iloc[-7:].copy()
        
        # Prepare data for heatmap
        heatmap_data = []
        
        # Define KPI thresholds
        kpi_thresholds = {
            "order_fill_rate": {"good": 95, "medium": 90, "bad": 85},
            "on_time_delivery": {"good": 90, "medium": 85, "bad": 80},
            "forecast_accuracy": {"good": 85, "medium": 80, "bad": 75},
            "inventory_turns": {"good": 8, "medium": 6, "bad": 4},
            "perfect_order": {"good": 90, "medium": 85, "bad": 80},
            "cash_to_cash": {"good": 40, "medium": 50, "bad": 60},
            "supply_chain_cost": {"good": 10, "medium": 12, "bad": 15}
        }
        
        # Create heatmap data
        for _, row in last_7_days.iterrows():
            for kpi in ["order_fill_rate", "on_time_delivery", "forecast_accuracy", "inventory_turns", "perfect_order"]:
                value = row[kpi]
                thresholds = kpi_thresholds[kpi]
                
                if value >= thresholds["good"]:
                    status = "Good"
                elif value >= thresholds["medium"]:
                    status = "Medium"
                else:
                    status = "Bad"
                
                heatmap_data.append({
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "kpi": kpi_display_names.get(kpi, kpi),
                    "value": value,
                    "status": status
                })
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create heatmap
        if not heatmap_df.empty:
            fig = px.density_heatmap(
                heatmap_df,
                x="date",
                y="kpi",
                z="value",
                color_continuous_scale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                title="KPI Performance Heatmap (Last 7 Days)"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="KPI",
                coloraxis_colorbar=dict(title="Value")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Alert Management Tab
    with tab2:
        st.markdown("### Alert Management")
        
        st.markdown("""
        Monitor and manage supply chain alerts and exceptions in real-time.
        """)
        
        # Alert filters
        col1, col2 = st.columns(2)
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity:",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High"]
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status:",
                ["New", "Acknowledged", "In Progress", "Resolved"],
                default=["New", "Acknowledged", "In Progress"]
            )
        
        # Apply filters
        filtered_alerts = alerts_df[
            alerts_df["severity"].isin(severity_filter) &
            alerts_df["status"].isin(status_filter)
        ]
        
        # Display alerts
        st.markdown("#### Active Alerts")
        
        if filtered_alerts.empty:
            st.info("No alerts match the selected filters.")
        else:
            # Color-code by severity
            def highlight_severity(val):
                color_map = {
                    "Critical": "background-color: #ff0000; color: white",
                    "High": "background-color: #ff9900; color: white",
                    "Medium": "background-color: #ffcc00",
                    "Low": "background-color: #99cc00"
                }
                return color_map.get(val, "")
            
            # Display styled dataframe
            st.dataframe(
                filtered_alerts.style.applymap(
                    highlight_severity, 
                    subset=["severity"]
                )
            )
        
        # Alert statistics
        st.markdown("#### Alert Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count alerts by severity
            severity_counts = alerts_df["severity"].value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            
            fig = px.bar(
                severity_counts,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map={
                    "Critical": "red",
                    "High": "orange",
                    "Medium": "yellow",
                    "Low": "green"
                },
                title="Alerts by Severity"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Count alerts by type
            type_counts = alerts_df["type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            
            fig = px.pie(
                type_counts,
                values="Count",
                names="Type",
                title="Alerts by Type"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Alert response time
        st.markdown("#### Alert Response Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Response Time", 
                "45 minutes"
            )
        
        with col2:
            st.metric(
                "Resolution Rate", 
                "87%"
            )
        
        with col3:
            st.metric(
                "Alerts Requiring Escalation", 
                "12%"
            )
    
    # Process Monitoring Tab
    with tab3:
        st.markdown("### Process Monitoring")
        
        st.markdown("""
        Monitor key business processes across your supply chain to identify bottlenecks and improvement opportunities.
        """)
        
        # Process selection
        selected_process = st.selectbox(
            "Select Process to Monitor:",
            ["Order to Delivery", "Forecast to Plan", "Procure to Pay", "Plan to Produce", "Inventory Management"]
        )
        
        # Order to Delivery Process
        if selected_process == "Order to Delivery":
            st.markdown("#### Order to Delivery Process")
            
            # Process metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Order Processing Time", 
                    "1.2 days",
                    "-0.3 days"
                )
            
            with col2:
                st.metric(
                    "Warehouse Fulfillment Time", 
                    "0.8 days",
                    "-0.1 days"
                )
            
            with col3:
                st.metric(
                    "Transit Time", 
                    "3.5 days",
                    "+0.2 days",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Total Lead Time", 
                    "5.5 days",
                    "-0.2 days"
                )
            
            # Process flow visualization
            st.markdown("#### Process Flow")
            
            # Sample process flow data
            process_stages = ["Order Receipt", "Credit Check", "Allocation", "Picking", "Packing", "Shipping", "Delivery"]
            stage_times = [0.5, 0.3, 0.4, 0.5, 0.3, 2.0, 1.5]  # in days
            bottleneck_threshold = 0.8  # 80% of max time
            
            # Create a horizontal bar chart for process flow
            fig = go.Figure()
            
            # Determine bottlenecks
            max_time = max(stage_times)
            bottleneck_threshold_value = max_time * bottleneck_threshold
            
            # Add bars for each stage
            for i, (stage, time) in enumerate(zip(process_stages, stage_times)):
                # Color coding: red for bottlenecks, blue for others
                color = "red" if time >= bottleneck_threshold_value else "blue"
                
                fig.add_trace(go.Bar(
                    y=[stage],
                    x=[time],
                    orientation="h",
                    name=stage,
                    marker_color=color,
                    text=[f"{time} days"],
                    textposition="inside"
                ))
            
            # Update layout
            fig.update_layout(
                title="Order to Delivery Process Flow and Duration",
                xaxis_title="Duration (days)",
                yaxis=dict(
                    title="Process Stage",
                    categoryorder="array",
                    categoryarray=process_stages[::-1]  # Reverse order for logical flow
                ),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Process performance over time
            st.markdown("#### Process Performance Trend")
            
            # Generate some sample data for process performance
            dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
            
            # Total lead time with some variability and slight improvement
            lead_times = [5.5 + 0.8 * np.sin(i/15) - i/300 + np.random.uniform(-0.3, 0.3) for i in range(90)]
            
            # Create DataFrame
            lead_time_df = pd.DataFrame({
                "date": dates,
                "lead_time": lead_times
            })
            
            # Add target line
            target = 5.0
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=lead_time_df["date"],
                y=lead_time_df["lead_time"],
                mode="lines",
                name="Lead Time",
                line=dict(color="blue")
            ))
            
            fig.add_trace(go.Scatter(
                x=lead_time_df["date"],
                y=[target] * len(lead_time_df),
                mode="lines",
                name="Target",
                line=dict(color="green", dash="dash")
            ))
            
            fig.update_layout(
                title="Order to Delivery Lead Time Trend (Last 90 Days)",
                xaxis_title="Date",
                yaxis_title="Lead Time (days)",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Process exceptions
            st.markdown("#### Process Exceptions")
            
            # Sample exception data
            exceptions = [
                {"order_id": "ORD-12345", "stage": "Credit Check", "issue": "Credit limit exceeded", "delay": "1.2 days"},
                {"order_id": "ORD-23456", "stage": "Allocation", "issue": "Inventory shortage", "delay": "2.5 days"},
                {"order_id": "ORD-34567", "stage": "Shipping", "issue": "Carrier delay", "delay": "1.8 days"},
                {"order_id": "ORD-45678", "stage": "Delivery", "issue": "Weather disruption", "delay": "2.0 days"},
            ]
            
            st.table(pd.DataFrame(exceptions))
        
        # Other processes would have similar layouts but different metrics and visualizations
        elif selected_process in ["Forecast to Plan", "Procure to Pay", "Plan to Produce", "Inventory Management"]:
            st.info(f"Detailed {selected_process} process monitoring will be available in future updates.")
    
    # Supply Chain Visibility Tab
    with tab4:
        st.markdown("### Supply Chain Visibility")
        
        st.markdown("""
        Gain visibility across your end-to-end supply chain, from suppliers to customers.
        """)
        
        # View selection
        view_type = st.selectbox(
            "Select View:",
            ["Network Overview", "Inventory Status", "Transportation Status", "Manufacturing Status"]
        )
        
        # Network Overview
        if view_type == "Network Overview":
            st.markdown("#### Supply Chain Network Overview")
            
            # Network health indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Network Health", 
                    "86%",
                    "-2%",
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Supplier Health", 
                    "82%",
                    "-5%",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Manufacturing Health", 
                    "91%",
                    "+1%"
                )
            
            with col4:
                st.metric(
                    "Distribution Health", 
                    "87%",
                    "-1%",
                    delta_color="inverse"
                )
            
            # Network map placeholder
            st.markdown("#### Interactive Supply Chain Map")
            st.info("Interactive supply chain network visualization will be available in future updates.")
            
            # Sample image placeholder for supply chain network
            st.markdown("""
            ```
            [Supplier] --> [Raw Materials] --> [Manufacturing] --> [Distribution Centers] --> [Retailers] --> [Customers]
                                   |                  |                     |
                              [Inventory]        [Inventory]           [Inventory]
            ```
            """)
            
            # Risk indicators
            st.markdown("#### Supply Chain Risk Indicators")
            
            # Sample risk data
            risk_data = {
                "category": ["Supplier", "Manufacturing", "Distribution", "Demand", "Environmental"],
                "risk_score": [72, 85, 67, 58, 90]
            }
            
            risk_df = pd.DataFrame(risk_data)
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=risk_df["category"],
                y=risk_df["risk_score"],
                marker_color=["red" if score < 70 else "orange" if score < 80 else "green" for score in risk_df["risk_score"]]
            ))
            
            fig.update_layout(
                title="Supply Chain Risk Assessment",
                xaxis_title="Category",
                yaxis_title="Risk Score (Higher is Better)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Inventory Status
        elif view_type == "Inventory Status":
            st.markdown("#### Inventory Status Overview")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Inventory Value", 
                    "$12.8M",
                    "+$0.3M",
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Inventory Days of Supply", 
                    "32 days",
                    "+2 days",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Stockout Rate", 
                    "2.3%",
                    "-0.5%"
                )
            
            # Inventory by location chart
            st.markdown("#### Inventory by Location")
            
            # Sample data
            inventory_by_location = {
                "location": ["East DC", "West DC", "Central DC", "North DC", "South DC"],
                "inventory_value": [3.2, 4.1, 2.8, 1.5, 1.2]
            }
            
            inv_loc_df = pd.DataFrame(inventory_by_location)
            
            fig = px.pie(
                inv_loc_df,
                values="inventory_value",
                names="location",
                title="Inventory Value by Location ($M)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Inventory health visualization
            st.markdown("#### Inventory Health Status")
            
            # Sample data for inventory health
            inventory_status = {
                "category": ["Raw Materials", "WIP", "Finished Goods", "Spare Parts"],
                "healthy": [75, 60, 80, 90],
                "at_risk": [15, 30, 12, 7],
                "critical": [10, 10, 8, 3]
            }
            
            # Convert to long format for stacked bar chart
            inventory_status_long = pd.DataFrame({
                "Category": np.repeat(inventory_status["category"], 3),
                "Status": np.tile(["Healthy", "At Risk", "Critical"], 4),
                "Percentage": np.concatenate([
                    inventory_status["healthy"],
                    inventory_status["at_risk"],
                    inventory_status["critical"]
                ])
            })
            
            # Create stacked bar chart
            fig = px.bar(
                inventory_status_long,
                x="Category",
                y="Percentage",
                color="Status",
                color_discrete_map={
                    "Healthy": "green",
                    "At Risk": "orange",
                    "Critical": "red"
                },
                title="Inventory Health by Category"
            )
            
            fig.update_layout(
                xaxis_title="Category",
                yaxis_title="Percentage",
                yaxis=dict(range=[0, 100]),
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Other views would have similar layouts but different metrics and visualizations
        elif view_type in ["Transportation Status", "Manufacturing Status"]:
            st.info(f"Detailed {view_type} visibility will be available in future updates.")
