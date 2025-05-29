"""Script to fix the indentation error in forecasting.py"""

import re

# Path to the forecasting.py file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\utils\forecasting.py"

# Read the current content
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the indentation error by finding and replacing the problematic pattern
# Look for the pattern with incorrect indentation
pattern = r'(\s+# Get forecast and map to custom dates\n\s+)forecast_values = model_fit\.forecast\(steps=forecast_periods\)\n\s+forecast_series = pd\.Series\(forecast_values\.values, index=future_index\)'

# Check if the pattern exists
if re.search(pattern, content):
    # Replace with correctly indented code
    fixed_content = re.sub(pattern, 
                          r'\1forecast_values = model_fit.forecast(steps=forecast_periods)\n            forecast_series = pd.Series(forecast_values, index=future_index)', 
                          content)
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed indentation error in forecasting.py")
else:
    print("Could not find the indentation error pattern. No changes made.")
