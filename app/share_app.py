"""
Enhanced IBP Sharing App
This script creates a shareable version of the IBP app
"""

import streamlit as st
import os
import socket
import webbrowser
from urllib.parse import urlparse
import requests
import sys
import time

# Add app directory to path
sys.path.append(os.path.dirname(__file__))

# Import from main app
from modules.demand.ui import show_demand_planning
from modules.inventory.ui import show_inventory_optimization
from modules.supply.ui import show_supply_planning
from modules.sop.ui import show_sop_alignment
from modules.response.ui import show_response_planning
from modules.control_tower.ui import show_control_tower

# Configure page
st.set_page_config(
    page_title="IBP System",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define modules
MODULES = {
    "📊 Demand Planning": show_demand_planning,
    "📦 Inventory Optimization": show_inventory_optimization,
    "⚙️ Supply Planning": show_supply_planning,
    "🔄 S&OP": show_sop_alignment,
    "🚨 Response Planning": show_response_planning,
    "👁️ Control Tower": show_control_tower,
}

def get_public_ip():
    """Get public IP address"""
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        return response.json()['ip']
    except:
        return "Unable to determine"

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "Unable to determine"

def get_sharing_info():
    """Get information for sharing the app"""
    # Get server URL
    server_url = st.experimental_get_query_params().get("server_url", [""])[0]
    if not server_url:
        server_url = urlparse(st.get_option("server.baseUrlPath")).netloc
    
    # Get port
    port = st.get_option("server.port") or 8510
    
    # Get IP addresses
    local_ip = get_local_ip()
    public_ip = get_public_ip()
    
    return {
        "local_url": f"http://localhost:{port}",
        "network_url": f"http://{local_ip}:{port}",
        "public_ip": public_ip,
        "public_url": f"http://{public_ip}:{port}",
        "port": port
    }

def show_sharing_sidebar():
    """Show sharing information in sidebar"""
    st.sidebar.title("IBP System")
    
    # App selection
    selected_module = st.sidebar.radio("Module", list(MODULES.keys()))
    
    # Separator
    st.sidebar.markdown("---")
    
    # Sharing information
    st.sidebar.markdown("### Sharing Information")
    info = get_sharing_info()
    
    st.sidebar.markdown("**Local Access:**")
    st.sidebar.markdown(f"🔗 [Open Locally]({info['local_url']})")
    
    st.sidebar.markdown("**Network Access:**")
    st.sidebar.markdown(f"🔗 [Open on Network]({info['network_url']})")
    
    st.sidebar.markdown("**Public Access:** *(requires port forwarding)*")
    st.sidebar.markdown(f"Public IP: `{info['public_ip']}`")
    st.sidebar.markdown(f"Port: `{info['port']}`")
    
    st.sidebar.warning(
        "For your friend to access from outside your network, "
        "set up port forwarding on your router:\n\n"
        f"External Port: `{info['port']}` → Internal: `{info['network_url']}`"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### App Information")
    st.sidebar.info(
        "This IBP System includes:\n"
        "- Multi-currency support (€, د.ت, $, etc.)\n"
        "- Enhanced forecast visualizations\n"
        "- Inventory optimization tools\n"
    )
    
    return selected_module

def main():
    """Main function"""
    selected_module = show_sharing_sidebar()
    
    # Display selected module
    MODULES[selected_module]()

if __name__ == "__main__":
    main()
