"""
Comprehensive script to fix ui.py by identifying and fixing the problematic try-except blocks.
"""
import re
import os
import shutil

def fix_syntax_errors():
    ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    backup_path = ui_path + ".complete-backup"
    
    # Create backup
    if not os.path.exists(backup_path):
        shutil.copy2(ui_path, backup_path)
        print(f"Created backup at {backup_path}")
    
    # Look for known problematic patterns
    try:
        with open(ui_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Recreate the file with fixed indentation
        with open(ui_path, 'w', encoding='utf-8') as f:
            in_problem_area = False
            skip_until_line = -1
            
            for i, line in enumerate(lines):
                line_num = i + 1
                
                # Skip lines we've already handled
                if line_num <= skip_until_line:
                    continue
                
                # Check for ARIMA Fallback dictionary that might be unclosed
                if "forecasts[\"ARIMA (Fallback)\"]" in line:
                    # This is where problems often start - handle this section specially
                    f.write('''                                forecasts["ARIMA (Fallback)"] = {
                                    'forecast': forecast,
                                    'model': 'Simple ARIMA Fallback',
                                    'error': str(e)
                                }
                            except Exception as fallback_error:
                                st.error(f"Even fallback forecast failed: {fallback_error}")
                                
        # If we have a DatetimeIndex, use our improved frequency detection
        try:
            if isinstance(train_data.index, pd.DatetimeIndex):    
                data_freq = get_data_frequency(train_data.index)
                # Store it for future use
                st.session_state['detected_frequency'] = data_freq
            else:
                # Default if all else fails
                data_freq = 'MS'
        except Exception as e:
            st.warning(f"Error detecting frequency: {str(e)}. Using default frequency.")
            data_freq = 'MS'
        seasonal_order = None
        confidence_interval = 0.95
        
''')
                    # Find the next proper section after this problematic area
                    for j in range(i+1, len(lines)):
                        if "st.info(f\"Using frequency: {data_freq}" in lines[j]:
                            # We found where to continue
                            skip_until_line = j
                            in_problem_area = False
                            break
                    # Continue with normal writing
                else:
                    # Write the line normally
                    f.write(line)
                    
        print(f"Fixed syntax errors in {ui_path}")
        print("The script focused on fixing the problematic ARIMA fallback and frequency detection sections.")
    except Exception as e:
        print(f"Error fixing file: {str(e)}")
        print("If you continue to have issues, you might need to manually fix the file.")

if __name__ == "__main__":
    fix_syntax_errors()
