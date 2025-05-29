import os
import re

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".analyze-fix-backup"

# Create a backup
if not os.path.exists(backup_path):
    with open(ui_path, 'r', encoding='utf-8') as src:
        content = src.read()
    with open(backup_path, 'w', encoding='utf-8') as dst:
        dst.write(content)
    print(f"Created backup at {backup_path}")

# Read the file
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific lines causing the issue (around line 2692-2710)
# This will completely replace the problematic section with properly structured code

# First, find the location where the error occurs
# We'll look for known patterns before and after the problematic section
search_pattern = re.compile(r"forecasts\[\"ARIMA \(Fallback\)\"\] = \{[^}]*'error': str\(e\)[^}]*\}.*?seasonal_order = None", re.DOTALL)
match = search_pattern.search(content)

if match:
    print(f"Found problematic section at position {match.start()}-{match.end()}")
    
    # Replace the entire section with a clean implementation
    replacement = """forecasts["ARIMA (Fallback)"] = {
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
        except Exception as e:
            st.warning(f"Error detecting frequency: {str(e)}. Using default frequency.")
            data_freq = 'MS'
        
        seasonal_order = None"""
    
    fixed_content = content[:match.start()] + replacement + content[match.end():]
    
    # Save the corrected file
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed problematic section in {ui_path}")
else:
    print("Could not find the problematic section. Manual fix may be required.")

print("Try running the application now.")
