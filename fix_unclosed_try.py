import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".unclosed-try-fix"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the structure for lines 245-254
fixed_section = [
    "                                        forecast = pd.Series([last_value] * forecast_periods)\n",
    "                                    \n",
    "                                    forecasts[\"ARIMA (Fallback)\"] = {\n",
    "                                        'forecast': forecast,\n",
    "                                        'model': 'Simple ARIMA Fallback',\n",
    "                                        'error': str(e)\n",
    "                                    }\n",
    "                                except Exception as fallback_error:\n",
    "                                    st.error(f\"Even fallback forecast failed: {fallback_error}\")\n",
    "\n"
]

# Replace the section from lines 245-253 with our fixed section
start_line = 245 - 1  # Convert to 0-indexed
end_line = 253 - 1    # Convert to 0-indexed

lines[start_line:end_line+1] = fixed_section

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed unclosed try block at lines 245-253 in {ui_path}")
print("Try running the application now.")
