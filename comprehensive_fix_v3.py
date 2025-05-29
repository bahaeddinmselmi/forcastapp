import os
import re

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".comprehensive-fix-v3"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the file content
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic section with a completely rewritten version
# The section includes the ARIMA forecast fallback code with proper indentation
start_marker = "                            try:"
end_marker = "                data_freq = 'MS'  # Default value"

# Find the start and end positions in the file
start_pos = content.find(start_marker)
end_pos = content.find(end_marker, start_pos) + len(end_marker)

if start_pos != -1 and end_pos != -1:
    # Create a proper replacement with correct indentation
    replacement = """                            try:
                                # Use safe helper to get target data
                                target_data = safely_get_target_data(train_data, target_col)
                                last_value = target_data.iloc[-1]
                                if future_index is not None:
                                    forecast = pd.Series([last_value] * len(future_index), index=future_index)
                                else:
                                    forecast = pd.Series([last_value] * forecast_periods)
                                
                                forecasts["ARIMA (Fallback)"] = {
                                    'forecast': forecast,
                                    'model': 'Simple ARIMA Fallback',
                                    'error': str(e)
                                }
                            except Exception as fallback_error:
                                st.error(f"Even fallback forecast failed: {fallback_error}")
                
                # After all forecasting attempts, detect the data frequency
                # If we have a DatetimeIndex, use our improved frequency detection
                data_freq = 'MS'  # Default value"""
    
    # Replace the problematic section
    new_content = content[:start_pos] + replacement + content[end_pos:]
    
    # Save the corrected file
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Fixed indentation and structure issues in the ARIMA fallback section of {ui_path}")
else:
    print("Could not find the target section. Please check the file manually.")

print("Try running the application now.")
