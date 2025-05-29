import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the app directory to the path
sys.path.append('C:\\Users\\Public\\Downloads\\ibp\\dd\\app')

# Import the forecasting functions
from utils.forecasting import (
    arima_forecast,
    exp_smoothing_forecast,
    xgboost_forecast,
    auto_arima_forecast,
    ensemble_forecast
)

# Create a simple time series dataset
def create_test_data(periods=24):
    dates = pd.date_range(start='2022-01-01', periods=periods, freq='MS')
    # Create a series with trend and seasonality
    values = np.arange(periods) * 5 + np.sin(np.arange(periods) * (2 * np.pi / 12)) * 20 + 100
    # Add some noise
    values += np.random.normal(0, 10, periods)
    return pd.Series(values, index=dates, name='value')

# Generate test data
data = create_test_data(24)
print("Test data:")
print(data.head())

# Set forecast periods
forecast_periods = 10

# Create future index for testing
last_date = data.index[-1]
future_index = pd.date_range(start=last_date, periods=forecast_periods+1, freq='MS')[1:]

# Test each forecasting function
print("\nTesting ARIMA forecast...")
arima_result = arima_forecast(data, periods=forecast_periods, future_index=future_index)
print(f"ARIMA forecast periods: {len(arima_result['forecast'])}")
print(arima_result['forecast'])

print("\nTesting Exponential Smoothing forecast...")
es_result = exp_smoothing_forecast(data, periods=forecast_periods, future_index=future_index)
print(f"Exponential Smoothing forecast periods: {len(es_result['forecast'])}")
print(es_result['forecast'])

print("\nTesting XGBoost forecast...")
# Create a DataFrame for XGBoost
data_df = pd.DataFrame({'value': data})
xgb_result = xgboost_forecast(data_df, periods=forecast_periods, target='value', future_index=future_index)
print(f"XGBoost forecast periods: {len(xgb_result['forecast'])}")
print(xgb_result['forecast'])

print("\nTesting Auto ARIMA forecast...")
auto_arima_result = auto_arima_forecast(data, periods=forecast_periods, future_index=future_index)
print(f"Auto ARIMA forecast periods: {len(auto_arima_result['forecast'])}")
print(auto_arima_result['forecast'])

print("\nTesting Ensemble forecast...")
ensemble_result = ensemble_forecast(data, periods=forecast_periods, future_index=future_index)
print(f"Ensemble forecast periods: {len(ensemble_result['forecast'])}")
print(ensemble_result['forecast'])

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(data.index, data, label='Historical Data', color='black')
plt.plot(arima_result['forecast'].index, arima_result['forecast'], label='ARIMA', linestyle='--')
plt.plot(es_result['forecast'].index, es_result['forecast'], label='Exponential Smoothing', linestyle='--')
plt.plot(xgb_result['forecast'].index, xgb_result['forecast'], label='XGBoost', linestyle='--')
plt.plot(auto_arima_result['forecast'].index, auto_arima_result['forecast'], label='Auto ARIMA', linestyle='--')
plt.plot(ensemble_result['forecast'].index, ensemble_result['forecast'], label='Ensemble', linestyle='--')
plt.legend()
plt.title('Improved Forecast Comparison')
plt.savefig('improved_forecast_comparison.png')
print("\nPlot saved as 'improved_forecast_comparison.png'")
