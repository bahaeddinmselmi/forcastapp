import os
import re

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".full-rewrite-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the entire file as text
with open(ui_path, 'r', encoding='utf-8') as f:
    file_content = f.read()

# Let's completely replace the problematic section
# We're looking for a section that starts with forecasting code and has try-except issues
pattern = re.compile(r'try:.*?# Use safe helper to get target data.*?Even fallback forecast failed:.*?data_freq = \'MS\'.*?except Exception as e:', re.DOTALL)
match = pattern.search(file_content)

if match:
    print(f"Found problematic section around position {match.start()}-{match.end()}")
    
    # Get some context before the match
    start_pos = max(0, match.start() - 500)
    end_pos = min(len(file_content), match.end() + 500)
    
    # Get content before and after our target section
    content_before = file_content[:match.start()]
    content_after = file_content[match.end():]
    
    # Create a properly structured replacement
    replacement = """try:
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

        # After all forecasting attempts, determine the data frequency
        data_freq = 'MS'  # Default value
        try:
            if isinstance(train_data.index, pd.DatetimeIndex):    
                data_freq = get_data_frequency(train_data.index)
                # Store it for future use
                st.session_state['detected_frequency'] = data_freq
            else:
                # Default if all else fails
                data_freq = 'MS'
        except Exception as e:"""
    
    # Combine everything back
    new_content = content_before + replacement + content_after
    
    # Save the corrected file
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Fixed syntax errors in {ui_path}")
else:
    print("Could not find the problematic section. Manual fix may be required.")

# Another common issue is indentation with else blocks
# Let's search for improperly indented else blocks
else_pattern = re.compile(r'([^\n]+)\n\s+else:', re.MULTILINE)
matches = else_pattern.finditer(file_content)

fixed_content = file_content
fixes_made = 0

# Process each problematic else
for match in matches:
    line_before = match.group(1)
    indent = re.match(r'(\s+)', line_before)
    if indent:
        # The else should have the same indentation as the line before
        proper_indent = indent.group(1)
        improper_else = match.group(0)
        proper_else = f"{line_before}\n{proper_indent}else:"
        fixed_content = fixed_content.replace(improper_else, proper_else)
        fixes_made += 1

if fixes_made > 0:
    # Save the fixed content
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print(f"Fixed {fixes_made} indentation issues with else blocks")

print("Try running the application now.")
