import streamlit as st
from pyngrok import ngrok
import os
import sys
import subprocess
import threading
import time

# Import existing main modules 
from modules.demand.ui import show_demand_planning
from modules.inventory.ui import show_inventory_optimization
from modules.supply.ui import show_supply_planning
from modules.sop.ui import show_sop_alignment
from modules.response.ui import show_response_planning
from modules.control_tower.ui import show_control_tower

# Define the navigation menu
PAGES = {
    "📊 Demand Planning": show_demand_planning,
    "📦 Inventory Optimization": show_inventory_optimization,
    "⚙️ Supply Planning": show_supply_planning,
    "🔄 S&OP": show_sop_alignment,
    "🚨 Response Planning": show_response_planning,
    "👁️ Control Tower": show_control_tower,
}

def run_ngrok():
    # Set up ngrok tunnel
    ngrok_tunnel = ngrok.connect(addr=8505, bind_tls=True)
    public_url = ngrok_tunnel.public_url
    print(f"\n\n✅ Your IBP app is publicly accessible at: {public_url}\n")
    # Keep the terminal link visible
    for _ in range(10):
        time.sleep(5)
        print(f"📣 Share this link with your friend: {public_url}")

def setup_page():
    st.set_page_config(
        page_title="Integrated Business Planning",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Add a logo and sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x70?text=IBP+System", width=150)
        st.title("Navigation")
        selection = st.radio("Go to", list(PAGES.keys()))
        
        # Add version information
        st.markdown("---")
        st.markdown("#### IBP System v1.0")
        st.markdown("Enhanced with multiple currency support")
        st.markdown("✅ Euro (€) - Default")
        st.markdown("✅ Tunisian Dinar (د.ت)")
        st.markdown("✅ USD, GBP, JPY, CNY")

    # Display the selected page
    PAGES[selection]()

def main():
    # Start ngrok in a separate thread
    ngrok_thread = threading.Thread(target=run_ngrok, daemon=True)
    ngrok_thread.start()
    
    # Set up the page
    setup_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
