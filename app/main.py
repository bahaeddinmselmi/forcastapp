"""
Main entry point for the Integrated Business Planning (IBP) Streamlit application.
"""

import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Integrated Business Planning",  # Will be replaced with config.APP_TITLE
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
from pathlib import Path

# Ensure app directory is in path
sys.path.append(str(Path(__file__).parent))

# Import configuration
import config

# Page title is set via st.set_page_config already

# Import module UI components
from modules.demand.ui import show_demand_planning
from modules.inventory.ui import show_inventory_optimization
from modules.supply.ui import show_supply_planning
from modules.sop.ui import show_sop_alignment
from modules.response.ui import show_response_planning
from modules.control_tower.ui import show_control_tower

def main():
    """
    Main function to run the Streamlit application.
    """
    # Add custom CSS
    with open(os.path.join(Path(__file__).parent, "static", "style.css"), "w") as f:
        f.write("""
        .main-header {
            font-size: 2.5rem;
            color: #0066B7;
        }
        .module-header {
            font-size: 1.8rem;
            color: #333333;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        """)

    # Create a directory for static files if it doesn't exist
    os.makedirs(os.path.join(Path(__file__).parent, "static"), exist_ok=True)
    
    # App title and description
    st.markdown(f"<h1 class='main-header'>{config.APP_TITLE}</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This integrated business planning system helps you align your demand forecasting, 
    inventory optimization, supply planning, and financial objectives into a cohesive strategy.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Module", list(config.PAGES.keys()))

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Integrated Business Planning System**
    
    Align your business processes from demand planning to 
    execution with our comprehensive IBP solution.
    
    Use the navigation above to switch between different
    planning modules.
    """)
    
    # Display selected page
    if page == "Demand Planning":
        show_demand_planning()
    elif page == "Inventory Optimization":
        show_inventory_optimization()
    elif page == "Supply Planning":
        show_supply_planning()
    elif page == "S&OP Alignment":
        show_sop_alignment()
    elif page == "Response Planning":
        show_response_planning()
    elif page == "Control Tower":
        show_control_tower()

if __name__ == "__main__":
    main()
