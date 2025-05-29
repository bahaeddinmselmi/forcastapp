"""
Script to fix syntax errors in ui.py by ensuring all try/except blocks are properly matched
and all indentation is correct.
"""
import os
import re
import shutil

def fix_ui_file():
    # Path to the ui.py file
    ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    backup_path = ui_path + ".bak"
    
    # Create backup
    shutil.copy2(ui_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(ui_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix specific syntax error: the unclosed dictionary at around line 2688
    # Pattern to match the dictionary with missing closing brace
    pattern1 = r"forecasts\[\"ARIMA \(Fallback\)\"\] = \{\s*'forecast': forecast,\s*'model': 'Simple ARIMA Fallback',\s*"
    replacement1 = """forecasts["ARIMA (Fallback)"] = {
                                    'forecast': forecast,
                                    'model': 'Simple ARIMA Fallback',
                                    'error': str(e)
                                }
                            except Exception as fallback_error:
                                st.error(f"Even fallback forecast failed: {fallback_error}")"""
    
    # Pattern for the frequency detection with improper indentation
    pattern2 = r"# If we have a DatetimeIndex, use our improved frequency detection\s*try:\s*if isinstance\(train_data\.index, pd\.DatetimeIndex\):\s*"
    replacement2 = """        # If we have a DatetimeIndex, use our improved frequency detection
        try:
            if isinstance(train_data.index, pd.DatetimeIndex):    
                """
    
    # Improper indentation in the else block
    pattern3 = r"else:\s*# Default if all else fails\s*data_freq = 'MS'"
    replacement3 = """            else:
                # Default if all else fails
                data_freq = 'MS'"""
    
    # Fix patterns
    content = re.sub(pattern1, replacement1, content)
    content = re.sub(pattern2, replacement2, content)
    content = re.sub(pattern3, replacement3, content)
    
    # Write the fixed content back
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed syntax errors in {ui_path}")
    print("If you still encounter issues, you may need to manually edit the file or use the backup.")

if __name__ == "__main__":
    fix_ui_file()
