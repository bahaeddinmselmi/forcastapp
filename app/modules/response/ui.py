"""
Response Planning module UI components.
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

def show_response_planning():
    """
    Display the Response Planning module interface.
    """
    st.markdown("## Supply Chain Response Planning")
    
    st.markdown("""
    The IBP for Response module helps you re-plan in real-time when supply chain disruptions occur.
    Simulate different scenarios and assess their impact on your supply chain.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "Disruption Scenarios", 
        "Impact Analysis", 
        "Response Strategies"
    ])
    
    # Disruption Scenarios Tab
    with tab1:
        st.markdown("### Disruption Scenarios")
        
        st.markdown("""
        Define potential supply chain disruptions to analyze their impact on your business.
        """)
        
        # Select disruption type
        disruption_type = st.selectbox(
            "Disruption Type:",
            ["Supply Delay", "Demand Spike", "Quality Issue", "Transportation Disruption", "Supplier Bankruptcy", "Natural Disaster"]
        )
        
        # Disruption parameters
        st.markdown("#### Disruption Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Start date
            start_date = st.date_input(
                "Disruption Start Date:",
                datetime.now() + timedelta(days=15)
            )
            
            # Duration
            duration = st.slider(
                "Duration (days):",
                min_value=1,
                max_value=90,
                value=14
            )
        
        with col2:
            # Impact severity
            severity = st.select_slider(
                "Impact Severity:",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium"
            )
            
            # Probability
            probability = st.slider(
                "Probability (%):",
                min_value=1,
                max_value=100,
                value=25
            )
        
        # Specific parameters based on disruption type
        st.markdown("#### Specific Parameters")
        
        if disruption_type == "Supply Delay":
            col1, col2 = st.columns(2)
            
            with col1:
                affected_supplier = st.selectbox(
                    "Affected Supplier:",
                    ["Supplier A", "Supplier B", "Supplier C", "All Suppliers"]
                )
            
            with col2:
                delivery_delay = st.slider(
                    "Delivery Delay (days):",
                    min_value=1,
                    max_value=60,
                    value=10
                )
            
            affected_materials = st.multiselect(
                "Affected Materials:",
                ["Aluminum", "Plastic", "Steel", "Electronics", "Packaging"]
            )
        
        elif disruption_type == "Demand Spike":
            col1, col2 = st.columns(2)
            
            with col1:
                demand_increase = st.slider(
                    "Demand Increase (%):",
                    min_value=10,
                    max_value=300,
                    value=50
                )
            
            with col2:
                affected_region = st.selectbox(
                    "Affected Region:",
                    ["North America", "Europe", "Asia", "Global"]
                )
            
            affected_products = st.multiselect(
                "Affected Products:",
                ["Premium Widget", "Standard Widget", "Economy Widget", "Deluxe Gadget", "Basic Gadget"]
            )
        
        elif disruption_type == "Quality Issue":
            col1, col2 = st.columns(2)
            
            with col1:
                defect_rate = st.slider(
                    "Defect Rate (%):",
                    min_value=1,
                    max_value=100,
                    value=15
                )
            
            with col2:
                affected_batch = st.text_input(
                    "Affected Batch Number:",
                    value="B2025-042"
                )
            
            affected_products = st.multiselect(
                "Affected Products:",
                ["Premium Widget", "Standard Widget", "Economy Widget", "Deluxe Gadget", "Basic Gadget"]
            )
        
        elif disruption_type == "Transportation Disruption":
            col1, col2 = st.columns(2)
            
            with col1:
                disruption_type = st.selectbox(
                    "Transportation Type:",
                    ["Sea Freight", "Air Freight", "Road Transport", "Rail Transport"]
                )
            
            with col2:
                route = st.selectbox(
                    "Affected Route:",
                    ["Asia to North America", "Europe to North America", "Intra-Europe", "Domestic"]
                )
            
            alternative_routes = st.multiselect(
                "Available Alternative Routes:",
                ["Alternative Sea Route", "Air Freight", "Rail Transport", "Multi-modal"]
            )
        
        # Save scenario
        if st.button("Save Disruption Scenario"):
            st.success("Disruption scenario saved for analysis!")
            
            # Store scenario information in session state
            scenario = {
                "disruption_type": disruption_type,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "duration": duration,
                "severity": severity,
                "probability": probability
            }
            
            # Add specific parameters based on disruption type
            if disruption_type == "Supply Delay":
                scenario.update({
                    "affected_supplier": affected_supplier,
                    "delivery_delay": delivery_delay,
                    "affected_materials": affected_materials
                })
            elif disruption_type == "Demand Spike":
                scenario.update({
                    "demand_increase": demand_increase,
                    "affected_region": affected_region,
                    "affected_products": affected_products
                })
            elif disruption_type == "Quality Issue":
                scenario.update({
                    "defect_rate": defect_rate,
                    "affected_batch": affected_batch,
                    "affected_products": affected_products
                })
            elif disruption_type == "Transportation Disruption":
                scenario.update({
                    "disruption_type": disruption_type,
                    "route": route,
                    "alternative_routes": alternative_routes
                })
            
            st.session_state['disruption_scenario'] = scenario
    
    # Impact Analysis Tab
    with tab2:
        st.markdown("### Impact Analysis")
        
        if 'disruption_scenario' not in st.session_state:
            st.info("Please define a disruption scenario in the Disruption Scenarios tab first.")
        else:
            st.markdown("""
            Analyze the potential impact of the defined disruption on your supply chain.
            """)
            
            # Show scenario summary
            scenario = st.session_state['disruption_scenario']
            
            st.markdown("#### Scenario Summary")
            st.write(f"**Disruption Type:** {scenario['disruption_type']}")
            st.write(f"**Start Date:** {scenario['start_date']}")
            st.write(f"**Duration:** {scenario['duration']} days")
            st.write(f"**Severity:** {scenario['severity']}")
            st.write(f"**Probability:** {scenario['probability']}%")
            
            # Run impact analysis
            if st.button("Run Impact Analysis"):
                with st.spinner("Analyzing disruption impact..."):
                    # Simulate impact analysis
                    
                    # Create time range for analysis
                    start = datetime.strptime(scenario['start_date'], "%Y-%m-%d")
                    dates = [start + timedelta(days=i) for i in range(scenario['duration'] + 30)]  # Extend 30 days beyond disruption
                    
                    # Create impact data based on disruption type
                    if scenario['disruption_type'] == "Supply Delay":
                        # Supply impact
                        baseline_supply = 100
                        disrupted_supply = []
                        
                        for i, date in enumerate(dates):
                            current_date = date.date()
                            scenario_start = datetime.strptime(scenario['start_date'], "%Y-%m-%d").date()
                            
                            if current_date < scenario_start:
                                # Before disruption
                                disrupted_supply.append(baseline_supply)
                            elif current_date < scenario_start + timedelta(days=scenario['duration']):
                                # During disruption
                                severity_factor = {"Low": 0.7, "Medium": 0.5, "High": 0.3, "Critical": 0.1}[scenario['severity']]
                                disrupted_supply.append(baseline_supply * severity_factor)
                            else:
                                # Recovery phase
                                days_since_end = (current_date - (scenario_start + timedelta(days=scenario['duration']))).days
                                recovery_rate = min(1.0, 0.5 + (days_since_end / 20))  # Gradually recover over 20 days
                                disrupted_supply.append(baseline_supply * recovery_rate)
                        
                        # Create DataFrame for visualization
                        impact_df = pd.DataFrame({
                            'date': dates,
                            'baseline_supply': baseline_supply,
                            'disrupted_supply': disrupted_supply
                        })
                        
                        # Plot impact
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=impact_df['date'],
                            y=impact_df['baseline_supply'],
                            mode='lines',
                            name='Baseline Supply',
                            line=dict(color='blue')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=impact_df['date'],
                            y=impact_df['disrupted_supply'],
                            mode='lines',
                            name='Disrupted Supply',
                            line=dict(color='red')
                        ))
                        
                        # Add shaded area for disruption period
                        disruption_end = datetime.strptime(scenario['start_date'], "%Y-%m-%d") + timedelta(days=scenario['duration'])
                        
                        fig.add_vrect(
                            x0=scenario['start_date'],
                            x1=disruption_end,
                            fillcolor="red",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )
                        
                        fig.update_layout(
                            title=f"Impact of {scenario['disruption_type']} on Supply",
                            xaxis_title="Date",
                            yaxis_title="Supply Level (%)",
                            legend=dict(x=0.01, y=0.99),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Impact metrics
                        total_lost_supply = sum([baseline_supply - ds for ds in disrupted_supply])
                        max_impact = min(disrupted_supply) / baseline_supply * 100
                        recovery_time = 0
                        
                        for i, val in enumerate(disrupted_supply):
                            if i >= scenario['duration'] and val >= 0.95 * baseline_supply:
                                recovery_time = i - scenario['duration']
                                break
                        
                        # Display impact metrics
                        st.markdown("#### Impact Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Lost Supply", f"{total_lost_supply:.0f} units")
                        
                        with col2:
                            st.metric("Maximum Impact", f"{100 - max_impact:.1f}% reduction")
                        
                        with col3:
                            st.metric("Recovery Time", f"{recovery_time} days")
                        
                        # Business impact assessment
                        st.markdown("#### Business Impact Assessment")
                        
                        # Calculate financial impact (simplified)
                        avg_unit_price = 35  # Assumed average price per unit
                        revenue_impact = total_lost_supply * avg_unit_price
                        
                        st.markdown(f"""
                        **Financial Impact:**
                        - Revenue Impact: ${revenue_impact:,.0f}
                        - Increased expediting costs: ${total_lost_supply * 10:,.0f}
                        - Total Financial Impact: ${revenue_impact + total_lost_supply * 10:,.0f}
                        
                        **Operational Impact:**
                        - Production line stoppages: {scenario['duration'] // 2} days
                        - Customer orders affected: ~{int(total_lost_supply / 10)} orders
                        - Service level impact: {int(100 - max_impact * 0.8)}% reduction
                        """)
                    
                    elif scenario['disruption_type'] == "Demand Spike":
                        # Create demand impact data
                        baseline_demand = 100
                        disrupted_demand = []
                        
                        for i, date in enumerate(dates):
                            current_date = date.date()
                            scenario_start = datetime.strptime(scenario['start_date'], "%Y-%m-%d").date()
                            
                            if current_date < scenario_start:
                                # Before disruption
                                disrupted_demand.append(baseline_demand)
                            elif current_date < scenario_start + timedelta(days=scenario['duration']):
                                # During disruption
                                spike_factor = 1 + (scenario['demand_increase'] / 100)
                                disrupted_demand.append(baseline_demand * spike_factor)
                            else:
                                # After disruption
                                days_since_end = (current_date - (scenario_start + timedelta(days=scenario['duration']))).days
                                recovery_rate = max(1.0, spike_factor - (days_since_end / 10) * (spike_factor - 1))
                                disrupted_demand.append(baseline_demand * recovery_rate)
                        
                        # Create supply capacity (fixed)
                        max_supply_capacity = baseline_demand * 1.2  # 20% buffer
                        
                        # Create DataFrame for visualization
                        impact_df = pd.DataFrame({
                            'date': dates,
                            'baseline_demand': baseline_demand,
                            'disrupted_demand': disrupted_demand,
                            'max_supply_capacity': max_supply_capacity
                        })
                        
                        # Plot impact
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=impact_df['date'],
                            y=impact_df['baseline_demand'],
                            mode='lines',
                            name='Baseline Demand',
                            line=dict(color='blue')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=impact_df['date'],
                            y=impact_df['disrupted_demand'],
                            mode='lines',
                            name='Spiked Demand',
                            line=dict(color='red')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=impact_df['date'],
                            y=impact_df['max_supply_capacity'],
                            mode='lines',
                            name='Max Supply Capacity',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        # Add shaded area for disruption period
                        disruption_end = datetime.strptime(scenario['start_date'], "%Y-%m-%d") + timedelta(days=scenario['duration'])
                        
                        fig.add_vrect(
                            x0=scenario['start_date'],
                            x1=disruption_end,
                            fillcolor="red",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )
                        
                        fig.update_layout(
                            title=f"Impact of {scenario['disruption_type']} on Demand",
                            xaxis_title="Date",
                            yaxis_title="Demand Level (%)",
                            legend=dict(x=0.01, y=0.99),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate impact metrics
                        unmet_demand = [max(0, d - max_supply_capacity) for d in disrupted_demand]
                        total_unmet_demand = sum(unmet_demand)
                        max_gap = max(unmet_demand)
                        
                        # Display impact metrics
                        st.markdown("#### Impact Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Unmet Demand", f"{total_unmet_demand:.0f} units")
                        
                        with col2:
                            st.metric("Peak Demand Increase", f"{scenario['demand_increase']}%")
                        
                        with col3:
                            st.metric("Max Supply Gap", f"{max_gap:.0f} units")
                        
                        # Business impact assessment
                        st.markdown("#### Business Impact Assessment")
                        
                        # Calculate financial impact
                        avg_unit_price = 35  # Assumed average price per unit
                        lost_revenue = total_unmet_demand * avg_unit_price
                        captured_revenue = sum([min(d, max_supply_capacity) - baseline_demand for d in disrupted_demand]) * avg_unit_price
                        
                        st.markdown(f"""
                        **Financial Impact:**
                        - Lost Revenue (Unmet Demand): ${lost_revenue:,.0f}
                        - Captured Additional Revenue: ${captured_revenue:,.0f}
                        - Net Financial Impact: ${captured_revenue - lost_revenue:,.0f}
                        
                        **Operational Impact:**
                        - Customer service level impacted: {int((total_unmet_demand / sum(disrupted_demand)) * 100)}%
                        - Stockouts expected: Yes
                        - Overtime required: {int(scenario['duration'] * 0.7)} days
                        """)
    
    # Response Strategies Tab
    with tab3:
        st.markdown("### Response Strategies")
        
        if 'disruption_scenario' not in st.session_state:
            st.info("Please define a disruption scenario in the Disruption Scenarios tab first.")
        else:
            st.markdown("""
            Develop and evaluate strategies to respond to the disruption.
            """)
            
            # Show scenario summary
            scenario = st.session_state['disruption_scenario']
            
            st.markdown("#### Scenario Summary")
            st.write(f"**Disruption Type:** {scenario['disruption_type']}")
            st.write(f"**Severity:** {scenario['severity']}")
            st.write(f"**Duration:** {scenario['duration']} days")
            
            # Strategy selection
            st.markdown("#### Response Strategy Options")
            
            # Different strategies based on disruption type
            if scenario['disruption_type'] == "Supply Delay":
                strategies = {
                    "Use Alternative Suppliers": st.checkbox("Use Alternative Suppliers", value=True),
                    "Expedite Shipping": st.checkbox("Expedite Shipping", value=True),
                    "Use Safety Stock": st.checkbox("Use Safety Stock", value=True),
                    "Adjust Production Schedule": st.checkbox("Adjust Production Schedule"),
                    "Customer Prioritization": st.checkbox("Customer Prioritization")
                }
                
                # Strategy details
                st.markdown("#### Strategy Details")
                
                if strategies["Use Alternative Suppliers"]:
                    st.markdown("##### Alternative Suppliers")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        alt_supplier = st.selectbox(
                            "Alternative Supplier:",
                            ["Supplier X", "Supplier Y", "Supplier Z"]
                        )
                    
                    with col2:
                        price_premium = st.slider(
                            "Price Premium (%):",
                            min_value=0,
                            max_value=100,
                            value=15
                        )
                    
                    lead_time = st.slider(
                        "Lead Time (days):",
                        min_value=1,
                        max_value=30,
                        value=5
                    )
                
                if strategies["Expedite Shipping"]:
                    st.markdown("##### Expedited Shipping")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        shipping_method = st.selectbox(
                            "Shipping Method:",
                            ["Air Freight", "Express Road", "Priority Rail"]
                        )
                    
                    with col2:
                        shipping_cost = st.slider(
                            "Additional Cost (%):",
                            min_value=20,
                            max_value=300,
                            value=100
                        )
                    
                    time_saved = st.slider(
                        "Time Saved (days):",
                        min_value=1,
                        max_value=20,
                        value=7
                    )
            
            elif scenario['disruption_type'] == "Demand Spike":
                strategies = {
                    "Increase Production": st.checkbox("Increase Production", value=True),
                    "Prioritize Customers": st.checkbox("Prioritize Customers", value=True),
                    "Adjust Pricing": st.checkbox("Adjust Pricing"),
                    "Source Additional Capacity": st.checkbox("Source Additional Capacity"),
                    "Allocate from Other Regions": st.checkbox("Allocate from Other Regions")
                }
                
                # Strategy details
                st.markdown("#### Strategy Details")
                
                if strategies["Increase Production"]:
                    st.markdown("##### Production Increase")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        overtime_hours = st.slider(
                            "Overtime Hours/Week:",
                            min_value=0,
                            max_value=40,
                            value=20
                        )
                    
                    with col2:
                        max_capacity_increase = st.slider(
                            "Maximum Capacity Increase (%):",
                            min_value=10,
                            max_value=100,
                            value=30
                        )
                    
                    additional_cost = st.slider(
                        "Additional Cost per Unit (%):",
                        min_value=5,
                        max_value=50,
                        value=25
                    )
                
                if strategies["Prioritize Customers"]:
                    st.markdown("##### Customer Prioritization")
                    
                    prioritization_method = st.selectbox(
                        "Prioritization Method:",
                        ["Strategic Value", "Profitability", "Order Size", "Order Date"]
                    )
                    
                    customer_segments = st.multiselect(
                        "High Priority Segments:",
                        ["Key Accounts", "Long-term Contracts", "High Margin", "Strategic Partners"],
                        default=["Key Accounts", "Strategic Partners"]
                    )
            
            # Run strategy simulation
            if st.button("Simulate Response Strategy"):
                with st.spinner("Simulating response strategies..."):
                    # Simplified simulation for demo purposes
                    
                    # Create time range for simulation
                    start = datetime.strptime(scenario['start_date'], "%Y-%m-%d")
                    dates = [start + timedelta(days=i) for i in range(scenario['duration'] + 30)]  # Extend 30 days beyond disruption
                    
                    # Create baseline impact (no response)
                    baseline_impact = []
                    normal_level = 100
                    
                    for i, date in enumerate(dates):
                        current_date = date.date()
                        scenario_start = datetime.strptime(scenario['start_date'], "%Y-%m-%d").date()
                        
                        if current_date < scenario_start:
                            # Before disruption
                            baseline_impact.append(normal_level)
                        elif current_date < scenario_start + timedelta(days=scenario['duration']):
                            # During disruption
                            severity_factor = {"Low": 0.7, "Medium": 0.5, "High": 0.3, "Critical": 0.1}[scenario['severity']]
                            baseline_impact.append(normal_level * severity_factor)
                        else:
                            # Recovery phase
                            days_since_end = (current_date - (scenario_start + timedelta(days=scenario['duration']))).days
                            recovery_rate = min(1.0, 0.5 + (days_since_end / 20))  # Gradually recover over 20 days
                            baseline_impact.append(normal_level * recovery_rate)
                    
                    # Create response strategy impact
                    response_impact = []
                    
                    # Simplified model: response strategies improve the impact
                    strategy_count = sum(strategies.values())
                    strategy_effectiveness = 0.1 * strategy_count  # Each strategy improves situation by 10%
                    
                    for i, baseline in enumerate(baseline_impact):
                        current_date = dates[i].date()
                        scenario_start = datetime.strptime(scenario['start_date'], "%Y-%m-%d").date()
                        
                        if current_date < scenario_start:
                            # Before disruption
                            response_impact.append(normal_level)
                        else:
                            # During and after disruption
                            improvement = (normal_level - baseline) * strategy_effectiveness
                            response_impact.append(min(normal_level, baseline + improvement))
                    
                    # Create DataFrame for visualization
                    impact_df = pd.DataFrame({
                        'date': dates,
                        'baseline_impact': baseline_impact,
                        'response_impact': response_impact,
                        'normal_level': normal_level
                    })
                    
                    # Plot impact
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=impact_df['date'],
                        y=impact_df['normal_level'],
                        mode='lines',
                        name='Normal Level',
                        line=dict(color='green')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=impact_df['date'],
                        y=impact_df['baseline_impact'],
                        mode='lines',
                        name='No Response (Baseline)',
                        line=dict(color='red')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=impact_df['date'],
                        y=impact_df['response_impact'],
                        mode='lines',
                        name='With Response Strategy',
                        line=dict(color='blue')
                    ))
                    
                    # Add shaded area for disruption period
                    disruption_end = datetime.strptime(scenario['start_date'], "%Y-%m-%d") + timedelta(days=scenario['duration'])
                    
                    fig.add_vrect(
                        x0=scenario['start_date'],
                        x1=disruption_end,
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    )
                    
                    fig.update_layout(
                        title="Impact of Response Strategy",
                        xaxis_title="Date",
                        yaxis_title="Service Level (%)",
                        legend=dict(x=0.01, y=0.99),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate strategy effectiveness metrics
                    baseline_area = sum(normal_level - level for level in baseline_impact)
                    response_area = sum(normal_level - level for level in response_impact)
                    improvement = (baseline_area - response_area) / baseline_area * 100
                    
                    # Display impact metrics
                    st.markdown("#### Strategy Effectiveness")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Impact Reduction", f"{improvement:.1f}%")
                    
                    with col2:
                        st.metric("Recovery Time Improvement", f"{int(improvement / 5)} days")
                    
                    with col3:
                        cost_factor = sum(1 for s in strategies.values() if s) * 0.5
                        st.metric("Response Cost Factor", f"{cost_factor:.1f}x")
                    
                    # Strategy recommendation
                    st.markdown("#### Strategy Recommendation")
                    
                    st.info(f"""
                    Based on the simulation, the selected response strategies will reduce disruption impact by {improvement:.1f}%.
                    
                    **Key actions to implement:**
                    1. Activate the response strategies immediately when the disruption occurs
                    2. Monitor the effectiveness of the response daily
                    3. Be prepared to adjust the strategy if the actual disruption differs from the scenario
                    
                    **Expected results:**
                    - Service level will be maintained at {min(response_impact):.1f}% or better
                    - Total impact reduced by {improvement:.1f}%
                    - Response cost estimated at ${cost_factor * 10000:,.0f}
                    """)
                    
                    # Risk assessment
                    st.markdown("#### Risk Assessment")
                    
                    st.warning(f"""
                    **Remaining risks:**
                    - Response effectiveness may be lower than expected
                    - Disruption duration might extend beyond {scenario['duration']} days
                    - Secondary impacts not captured in the simulation
                    
                    **Contingency plan:**
                    - Prepare additional response options if service level drops below {min(response_impact) - 10:.1f}%
                    - Set up daily monitoring and response adjustment process
                    """)
