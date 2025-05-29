"""Script to manually fix the indentation in the exp_smoothing_forecast function"""

# Path to the file
file_path = "C:\\Users\\Public\\Downloads\\ibp\\dd\\app\\utils\\forecasting.py"

# Read the current content
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the exp_smoothing_forecast function
start_line = -1
end_line = -1
for i, line in enumerate(lines):
    if "def exp_smoothing_forecast" in line:
        start_line = i
        break

if start_line >= 0:
    # Find the problematic lines
    for i in range(start_line, len(lines)):
        if "forecast_values = model_fit.forecast(steps=forecast_periods)" in lines[i]:
            # Fix the indentation of this line
            lines[i] = "            forecast_values = model_fit.forecast(steps=forecast_periods)\n"
            
            # Also fix the next line which is likely the forecast_series line
            if i+1 < len(lines) and "forecast_series = pd.Series" in lines[i+1]:
                lines[i+1] = "            forecast_series = pd.Series(forecast_values.values, index=future_index)\n"
            
            print(f"Fixed indentation at lines {i} and {i+1}")
            break

    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Successfully fixed indentation in {file_path}")
else:
    print("Could not find the exp_smoothing_forecast function")
