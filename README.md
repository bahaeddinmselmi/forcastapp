# Integrated Business Planning (IBP) System

A modular Integrated Business Planning software system that helps align demand, inventory, supply, and financial planning processes.

## Features

- **IBP for Demand**: Short/mid-term demand forecasting using statistical models and ML algorithms
- **IBP for Inventory**: Stock level optimization using EOQ models and service-level metrics
- **IBP for Supply**: Production and supply planning based on capacity constraints
- **IBP for S&OP**: Strategic alignment of supply/demand planning with financial goals
- **IBP for Response**: Real-time re-planning for supply chain disruptions
- **IBP Control Tower**: End-to-end visibility of supply chain performance metrics

## Technical Stack

- **Backend**: Python with FastAPI
- **Frontend**: Streamlit for interactive dashboards
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Forecasting**: ARIMA, XGBoost, Prophet
- **Visualization**: Plotly, Matplotlib, Seaborn
- **ML Monitoring**: MLflow, Evidently AI, Deepchecks

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   cd app
   streamlit run main.py
   ```

## Module Overview

### Demand Planning
Forecast demand using historical data, market intelligence, and AI/ML models.

### Inventory Optimization
Optimize stock levels, manage safety stock, and reduce inventory costs.

### Supply Planning
Plan production and procurement based on demand forecasts and capacity constraints.

### S&OP Alignment
Align operational planning with financial objectives and KPIs.

### Response Planning
Simulate and respond to supply chain disruptions with dynamic replanning.

### Control Tower
Monitor end-to-end supply chain performance with real-time dashboards and alerts.

## Data Sources

The system can import data from:
- CSV files
- SQL databases
- External APIs (economic indicators, market trends, etc.)

## Scenario Planning

Users can create and simulate various "what-if" scenarios:
- Demand spikes or drops
- Price changes and promotions
- Supply chain disruptions
- Production capacity changes
