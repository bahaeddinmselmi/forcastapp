"""Create a completely fixed version of forecasting.py"""

import os

# Path to the original file
original_path = "C:\\Users\\Public\\Downloads\\ibp\\dd\\app\\utils\\forecasting.py"

# Create a backup of the original file
backup_path = original_path + ".original"
if not os.path.exists(backup_path):
    with open(original_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    
    print(f"Created backup at {backup_path}")

# Define the warning suppression code to add at the top
warning_code = """
# Add warning suppression
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence common warnings that affect forecasting
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")
"""

# Read the backup file to get the original content
with open(backup_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the problematic section with indentation issues
problematic_section = """
        if future_index is not None:
            # User wants to forecast for specific future dates
            forecast_periods = len(future_index)
            custom_dates_info = f"Using custom date range starting from {future_index[0].strftime('%Y-%m-%d')}"
            
{{ ... }}
            # Get forecast and map to custom dates
            forecast_values = model_fit.forecast(steps=forecast_periods)
            forecast_series = pd.Series(forecast_values.values, index=future_index)
        else:"""

# Define the corrected section
corrected_section = """
        if future_index is not None:
            # User wants to forecast for specific future dates
            forecast_periods = len(future_index)
            custom_dates_info = f"Using custom date range starting from {future_index[0].strftime('%Y-%m-%d')}"
            
            # Get forecast and map to custom dates
            forecast_values = model_fit.forecast(steps=forecast_periods)
            forecast_series = pd.Series(forecast_values.values, index=future_index)
        else:"""

# Add warning imports after the Prophet import
prophet_import_pos = content.find("from prophet import Prophet")
next_line_pos = content.find("\n", prophet_import_pos)
content_with_warnings = content[:next_line_pos+1] + warning_code + content[next_line_pos+1:]

# Replace the problematic section with the corrected section
# We'll use a more generic approach to find and replace the indentation issue
import re

# Find all instances of the pattern with indentation issues
pattern = re.compile(r'(\s+if future_index is not None:.*?forecast_periods = len\(future_index\).*?custom_dates_info.*?# Get forecast and map to custom dates\s+)forecast_values = model_fit\.forecast\(steps=forecast_periods\)', re.DOTALL)
fixed_content = pattern.sub(r'\1            forecast_values = model_fit.forecast(steps=forecast_periods)', content_with_warnings)

# Also fix the next line with forecast_series
pattern2 = re.compile(r'(forecast_values = model_fit\.forecast\(steps=forecast_periods\).*?)forecast_series = pd\.Series\(forecast_values', re.DOTALL)
fixed_content = pattern2.sub(r'\1            forecast_series = pd.Series(forecast_values', fixed_content)

# Write the fixed content to the original file
with open(original_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Fixed forecasting.py with corrected indentation and warning suppression")
