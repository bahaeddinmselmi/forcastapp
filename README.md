# Integrated Business Planning (IBP) System

A modular Integrated Business Planning software system that helps align demand, inventory, supply, and financial planning processes.

## Features

- **IBP for Demand**: Short/mid-term demand forecasting using statistical models and ML algorithms
- **IBP for Inventory**: Stock level optimization using EOQ models and service-level metrics
- **IBP for Supply**: Production and supply planning based on capacity constraints
- **IBP for S&OP**: Strategic alignment of supply/demand planning with financial goals
- **IBP for Response**: Real-time re-planning for supply chain disruptions
- **IBP for Control Tower**: End-to-end visibility of supply chain performance metrics

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
- Advanced forecasting algorithms including ARIMA, Exponential Smoothing, and Prophet
- Automated model selection based on data characteristics
- Support for multiple time series frequencies (daily, weekly, monthly)
- Interactive visualization of forecast results
- Model performance metrics and validation

### Inventory Optimization
- Dynamic safety stock calculations
- EOQ (Economic Order Quantity) optimization
- Service level-based inventory planning
- ABC/XYZ analysis
- Inventory cost optimization

### Supply Planning
Plan production and procurement based on demand forecasts and capacity constraints.

### S&OP Alignment
Align operational planning with financial objectives and KPIs.

### Response Planning
Simulate and respond to supply chain disruptions with dynamic replanning.

### Control Tower
Monitor end-to-end supply chain performance with real-time dashboards and alerts.

## Data Sources

The system supports various data import methods:
- CSV files with custom formatting
- Excel files (.xlsx, .xls)
- SQL databases
- External APIs (economic indicators, market trends, etc.)

## Sample Data

The `app/data` directory contains sample datasets for testing:
- Sales history
- Inventory levels
- Supply chain data
- KPI metrics
- Alerts and notifications

## Recent Updates

- Improved forecasting accuracy with automated frequency detection
- Enhanced UI with better error handling and user feedback
- Streamlined data import process
- Better exception handling and error messages
- Cleaned up codebase and removed redundant files
- Updated dependency management

## Scenario Planning

Users can create and simulate various "what-if" scenarios:
- Demand spikes or drops
- Price changes and promotions
- Supply chain disruptions
- Production capacity changes
- Market trend impacts

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## Deployment

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/bahaeddinmselmi/forcastapp.git
   cd forcastapp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app/main.py
   ```

### Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Log in to [Streamlit Cloud](https://share.streamlit.io)
3. Create a new app and select your forked repository
4. Set the following:
   - Main file path: `app/main.py`
   - Python version: 3.13
5. Click "Deploy!"

## Configuration

The app can be configured through:
- `.streamlit/config.toml` for Streamlit settings
- `app/config.py` for application settings
