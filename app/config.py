"""
Configuration settings for the IBP application.
Contains constants, file paths, and other shared settings.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = os.path.join(ROOT_DIR, "app", "data")

# Sample data files
SAMPLE_SALES_DATA = os.path.join(DATA_DIR, "sample_sales_data.csv")
SAMPLE_INVENTORY_DATA = os.path.join(DATA_DIR, "sample_inventory_data.csv")
SAMPLE_SUPPLY_DATA = os.path.join(DATA_DIR, "sample_supply_data.csv")

# Forecasting parameters
FORECAST_PERIODS = 12  # Default forecast horizon in months
MAX_FORECAST_PERIODS = 24  # Maximum allowed forecast periods
SEASONAL_PERIODS = 12  # Assuming monthly data with yearly seasonality

# Inventory planning parameters
TARGET_SERVICE_LEVEL = 0.95  # Default target service level
HOLDING_COST_RATE = 0.25  # Default inventory holding cost (25% of item value)
ORDER_COST = 100  # Default fixed cost per order in currency units

# Supply planning parameters
DEFAULT_LEAD_TIME = 30  # Default lead time in days
CAPACITY_BUFFER = 0.10  # Default capacity buffer (10%)

# S&OP parameters
FINANCIAL_METRICS = ["Revenue", "COGS", "Gross Margin", "Inventory Value"]
ROLLING_PERIODS = 12  # Default S&OP planning horizon in months

# Response planning parameters
DISRUPTION_SCENARIOS = ["Supply Delay", "Demand Spike", "Quality Issue", "Transportation Disruption"]
RESPONSE_STRATEGIES = ["Expedite", "Alternative Sourcing", "Prioritize Customers", "Adjust Production"]

# Control Tower parameters
KPI_UPDATE_INTERVAL = 60 * 60  # Update KPIs every hour (in seconds)
ALERT_LEVELS = {
    "critical": 0.15,  # 15% deviation
    "warning": 0.10,   # 10% deviation
    "notice": 0.05     # 5% deviation
}

# Currency settings
CURRENCY_SETTINGS = {
    "USD": {"symbol": "$", "name": "US Dollar"},
    "EUR": {"symbol": "€", "name": "Euro", "default": True},  # Set Euro as default
    "GBP": {"symbol": "£", "name": "British Pound"},
    "TND": {"symbol": "د.ت", "name": "Tunisian Dinar"},  # Added Tunisian Dinar
    "JPY": {"symbol": "¥", "name": "Japanese Yen"},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan"}
}

# Default currency
DEFAULT_CURRENCY = next((code for code, details in CURRENCY_SETTINGS.items() if details.get("default")), "USD")

# API endpoints
API_PREFIX = "/api/v1"
API_ENDPOINTS = {
    "demand_forecast": f"{API_PREFIX}/demand/forecast",
    "inventory_optimize": f"{API_PREFIX}/inventory/optimize",
    "supply_plan": f"{API_PREFIX}/supply/plan",
    "sop_integrate": f"{API_PREFIX}/sop/integrate",
    "response_analyze": f"{API_PREFIX}/response/analyze",
    "control_tower_kpis": f"{API_PREFIX}/control_tower/kpis",
}

# Graph color palettes
COLOR_PALETTE = {
    "actual": "#1f77b4",  # Blue
    "forecast": "#ff7f0e",  # Orange
    "upper_bound": "#2ca02c",  # Green
    "lower_bound": "#d62728",  # Red
    "target": "#9467bd",  # Purple
    "inventory": "#8c564b",  # Brown
    "supply": "#e377c2",  # Pink
    "capacity": "#7f7f7f",  # Gray
}

# App display settings
APP_TITLE = "Integrated Business Planning System"
PAGES = {
    "Demand Planning": "demand",
    "Inventory Optimization": "inventory",
    "Supply Planning": "supply",
    "S&OP Alignment": "sop",
    "Response Planning": "response",
    "Control Tower": "control_tower",
}
