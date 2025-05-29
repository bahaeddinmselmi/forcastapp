import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".missing-brace-fix-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the missing brace issue at line 2689
fixed_section = [
    "                                     forecast = pd.Series([last_value] * len(future_index), index=future_index)\n",
    "                                 else:\n",
    "                                     forecast = pd.Series([last_value] * forecast_periods)\n",
    "                                     \n",
    "                                 forecasts[\"ARIMA (Fallback)\"] = {\n",  # Added the opening statement for the dictionary
    "                                     'forecast': forecast,\n",
    "                                     'model': 'Simple ARIMA Fallback',\n",
    "                                     'error': str(e)\n",
    "                                 }\n"
]

# Replace the section from lines 2685-2693 with our fixed section
start_line = 2685 - 1  # Convert to 0-indexed
end_line = 2693 - 1    # Convert to 0-indexed

lines[start_line:end_line+1] = fixed_section

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed missing brace for dictionary at line 2689 in {ui_path}")
print("Try running the application now.")
