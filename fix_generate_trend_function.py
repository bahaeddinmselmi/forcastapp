import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".function-fix-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8', errors='replace') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# The corrected function definition
corrected_function = '''def generate_trend_fallback_forecast(train_data, periods, future_index=None):
    """
    Generate a simple trend-based forecast as a fallback when models fail.
    Uses the last value and trend information from the training data.
    
    Args:
        train_data: Historical data as pandas Series
        periods: Number of periods to forecast
        future_index: Optional custom date index for forecast
        
    Returns:
        Pandas Series with the fallback forecast
    """
    # Use at least the last 6 points to calculate trend, or all points if fewer than 6
    n_points = min(6, len(train_data))
    
    if n_points <= 1:
        # If only one point, use it as a constant forecast
        last_value = train_data.iloc[-1]
        trend = 0
    else:
        # Calculate a simple linear trend
        last_points = train_data.iloc[-n_points:]
        last_value = last_points.iloc[-1]
        first_value = last_points.iloc[0]
        trend = (last_value - first_value) / (n_points - 1)
    
    # Generate forecast values with trend
    forecast_values = [last_value + trend * (i+1) for i in range(periods)]
    
    # Ensure no negative values for demand forecasting
    forecast_values = [max(0, v) for v in forecast_values]
    
    # Create a Series with the proper index
    if future_index is not None:
        forecast = pd.Series(forecast_values, index=future_index[:periods])
    else:
        forecast = pd.Series(forecast_values)
    
    return forecast'''

# Let's find the start and end of the function in the file
function_start_pattern = "def generate_trend_fallback_forecast("
function_end_pattern = "return forecast"

# Read the entire file as a string
with open(ui_path, 'r', encoding='utf-8', errors='replace') as f:
    file_content = f.read()

# Find the function in the file
start_index = file_content.find(function_start_pattern)
if start_index == -1:
    print("Could not find the generate_trend_fallback_forecast function.")
    exit(1)

# Find the end of the function (return statement plus some room for the closing line)
end_index = file_content.find(function_end_pattern, start_index)
if end_index == -1:
    print("Could not find the end of the generate_trend_fallback_forecast function.")
    exit(1)
end_index = file_content.find("\n", end_index) + 1  # Include the newline after return

# Replace the function with the corrected version
new_file_content = file_content[:start_index] + corrected_function + file_content[end_index:]

# Write the corrected content back to the file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.write(new_file_content)

print(f"Fixed the generate_trend_fallback_forecast function in {ui_path}")
print("Try running the application now.")
