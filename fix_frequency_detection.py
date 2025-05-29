"""
Script to fix the frequency detection section in ui.py.
This will replace just the problematic code section with properly indented code.
"""
import re

def fix_specific_section():
    file_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    
    # Read file line by line to identify and fix the issue precisely
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create a backup
    with open(file_path + '.backup', 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
    # Define the corrected section (for the frequency detection and ARIMA fallback)
    fallback_section = '''                                forecasts["ARIMA (Fallback)"] = {
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
'''
    
    # Find the problematic section
    start_marker = '''                                forecasts["ARIMA (Fallback)"] = {'''
    end_marker = '''        seasonal_order = None
        confidence_interval = 0.95'''
    
    # Initialize variables to track section
    start_index = -1
    end_index = -1
    
    # Find the start and end of the section
    for i, line in enumerate(lines):
        if start_marker in line and start_index == -1:
            start_index = i
        if end_marker in line and start_index != -1:
            end_index = i + 1  # Include the end marker line
            break
    
    if start_index != -1 and end_index != -1:
        # Replace the section
        print(f"Found problematic section from line {start_index+1} to {end_index}")
        new_lines = lines[:start_index] + [fallback_section] + lines[end_index:]
        
        # Write the fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Fixed the frequency detection section in {file_path}")
    else:
        print("Could not locate the problematic section. No changes made.")

if __name__ == "__main__":
    fix_specific_section()
