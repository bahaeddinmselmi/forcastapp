import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".complete-fix-backup"

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

# Find the problematic section and completely replace it with properly structured code
fixed_section = [
    "                                forecasts[\"ARIMA (Fallback)\"] = {\n",
    "                                    'forecast': forecast,\n",
    "                                    'model': 'Simple ARIMA Fallback',\n",
    "                                    'error': str(e)\n",
    "                                }\n",
    "                            except Exception as fallback_error:\n",
    "                                st.error(f\"Even fallback forecast failed: {fallback_error}\")\n",
    "\n",
    "                # After all forecast attempts, determine the data frequency\n",
    "                data_freq = 'MS'  # Default value\n",
    "                try:\n",
    "                    if isinstance(train_data.index, pd.DatetimeIndex):    \n",
    "                        data_freq = get_data_frequency(train_data.index)\n",
    "                        # Store it for future use\n",
    "                        st.session_state['detected_frequency'] = data_freq\n",
    "                except Exception as e:\n",
    "                    st.warning(f\"Error detecting frequency: {str(e)}. Using default frequency.\")\n",
    "                    data_freq = 'MS'\n",
    "\n"
]

# Replace the section from lines 2690-2708 with our fixed section
start_line = 2690 - 1  # Convert to 0-indexed
end_line = 2708 - 1    # Convert to 0-indexed

lines[start_line:end_line+1] = fixed_section

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed structure for lines 2690-2708 in {ui_path}")
print("Try running the application now.")
