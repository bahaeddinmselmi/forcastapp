import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".try-except-fix-backup"

# Create a backup
if not os.path.exists(backup_path):
    with open(ui_path, 'r', encoding='utf-8') as src:
        content = src.read()
    with open(backup_path, 'w', encoding='utf-8') as dst:
        dst.write(content)
    print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Look for try-blocks without matching except blocks
# For this specific issue, we know around line 2697-2700 there's a problem
# Let's create a proper structure in this section

# Define the section we want to replace (lines 2680-2708)
# We'll make sure all try-except blocks are properly nested and closed
fixed_section = [
    "                            try:\n",
    "                                # Use safe helper to get target data\n",
    "                                target_data = safely_get_target_data(train_data, target_col)\n",
    "                                last_value = target_data.iloc[-1]\n",
    "                                if future_index is not None:\n",
    "                                    forecast = pd.Series([last_value] * len(future_index), index=future_index)\n",
    "                                else:\n",
    "                                    forecast = pd.Series([last_value] * forecast_periods)\n",
    "                                    \n",
    "                                forecasts[\"ARIMA (Fallback)\"] = {\n",
    "                                    'forecast': forecast,\n",
    "                                    'model': 'Simple ARIMA Fallback',\n",
    "                                    'error': str(e)\n",
    "                                }\n",
    "                            except Exception as fallback_error:\n",
    "                                st.error(f\"Even fallback forecast failed: {fallback_error}\")\n",
    "        \n",
    "        # After all forecasting attempts, determine the data frequency\n",
    "        data_freq = 'MS'  # Default value\n",
    "        try:\n",
    "            if isinstance(train_data.index, pd.DatetimeIndex):    \n",
    "                data_freq = get_data_frequency(train_data.index)\n",
    "                # Store it for future use\n",
    "                st.session_state['detected_frequency'] = data_freq\n",
    "        except Exception as e:\n",
    "            st.warning(f\"Error detecting frequency: {str(e)}. Using default frequency.\")\n",
    "            data_freq = 'MS'\n",
    "        \n",
    "        seasonal_order = None\n"
]

# Replace the section from lines 2680-2708 with our fixed section
start_line = 2680 - 1  # Convert to 0-indexed
end_line = 2708 - 1    # Convert to 0-indexed

lines[start_line:end_line+1] = fixed_section

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed try-except structure for lines 2680-2708 in {ui_path}")
print("Try running the application now.")
