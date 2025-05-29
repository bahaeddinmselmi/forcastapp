import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".indentation-line-240-fix"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the indentation at lines 239-245
fixed_section = [
    "                                try:\n",
    "                                    # Use safe helper to get target data\n",
    "                                    target_data = safely_get_target_data(train_data, target_col)\n",
    "                                    last_value = target_data.iloc[-1]\n",
    "                                    if future_index is not None:\n",
    "                                        forecast = pd.Series([last_value] * len(future_index), index=future_index)\n",
    "                                    else:\n",
    "                                        forecast = pd.Series([last_value] * forecast_periods)\n"
]

# Replace the section from lines 238-245 with our fixed section
start_line = 238 - 1  # Convert to 0-indexed
end_line = 245 - 1    # Convert to 0-indexed

lines[start_line:end_line+1] = fixed_section

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed indentation for try-except block at lines 238-245 in {ui_path}")
print("Try running the application now.")
