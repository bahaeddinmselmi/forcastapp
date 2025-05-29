"""
Quick script to fix the indentation issues in the Auto ARIMA implementation
"""

import re

def fix_indentation():
    file_path = r'C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.readlines()
    
    # Find the Auto ARIMA section that needs fixing
    start_line = None
    end_line = None
    in_auto_arima = False
    auto_arima_indent = None
    
    for i, line in enumerate(content):
        if "if 'Auto ARIMA' in models_to_run:" in line:
            in_auto_arima = True
            auto_arima_indent = len(line) - len(line.lstrip())
            start_line = i
        
        # Find the end of the Auto ARIMA section
        if in_auto_arima and "except Exception as e:" in line and "Error in Auto ARIMA" in content[i+1]:
            end_line = i + 2  # Include the error line
            break
    
    if start_line is not None and end_line is not None:
        # Extract the Auto ARIMA section
        auto_arima_section = content[start_line:end_line]
        
        # Fix the indentation
        fixed_section = []
        base_indent = " " * (auto_arima_indent + 4)  # Default indentation for this section
        
        for line in auto_arima_section:
            # First, remove all indentation
            stripped_line = line.lstrip()
            
            # Skip empty lines
            if not stripped_line:
                fixed_section.append("\n")
                continue
            
            # Determine the correct indentation
            if "if 'Auto ARIMA' in models_to_run:" in line:
                # This is the top-level line
                fixed_section.append(line)
            elif "try:" in stripped_line and "# Add more robust handling" in content[auto_arima_section.index(line)-1]:
                # This is the first try block
                fixed_section.append(" " * (auto_arima_indent + 4) + stripped_line)
            elif "with st.spinner" in stripped_line:
                # This is inside the first try block
                fixed_section.append(" " * (auto_arima_indent + 8) + stripped_line)
            elif "# Check if we can determine seasonality" in stripped_line:
                # This is inside the with block
                fixed_section.append(" " * (auto_arima_indent + 12) + stripped_line)
            elif "try:" in stripped_line and "# Call the advanced implementation" in content[auto_arima_section.index(line)-1]:
                # This is the inner try block
                fixed_section.append(" " * (auto_arima_indent + 12) + stripped_line)
            elif "except Exception as inner_e:" in stripped_line:
                # This is the inner except block
                fixed_section.append(" " * (auto_arima_indent + 12) + stripped_line)
            elif "except Exception as e:" in stripped_line:
                # This is the outer except block
                fixed_section.append(" " * (auto_arima_indent + 4) + stripped_line)
            else:
                # For all other lines, maintain consistent indentation within their blocks
                if "Error in Auto ARIMA model:" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "Create a basic fallback" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "# Make the max_seasonal_order" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "auto_arima_result =" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "if 'model' in auto_arima_result:" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                    # All sub-indents of this if
                elif "auto_arima_result['model'] =" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 20) + stripped_line)
                elif "# Verify forecast results are valid" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "if 'forecast' in auto_arima_result" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "if len(auto_arima_result['forecast'])" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 20) + stripped_line)
                elif "forecasts['Auto ARIMA'] =" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 24) + stripped_line)
                elif "if 'model_order' in auto_arima_result:" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 24) + stripped_line)
                elif "st.success(" in stripped_line and "Auto ARIMA selected order" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 28) + stripped_line)
                elif "else:" in stripped_line and len(stripped_line) < 10:
                    # Handle the nested else blocks
                    prev_line = auto_arima_section[auto_arima_section.index(line)-1]
                    if "if len(auto_arima_result['forecast'])" in prev_line:
                        fixed_section.append(" " * (auto_arima_indent + 20) + stripped_line)
                    elif "if 'forecast' in auto_arima_result" in prev_line:
                        fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                    else:
                        fixed_section.append(" " * (auto_arima_indent + 12) + stripped_line)
                elif "st.warning" in stripped_line:
                    prev_line = auto_arima_section[auto_arima_section.index(line)-1]
                    if "else:" in prev_line and len(prev_line.strip()) < 10:
                        if "if len(auto_arima_result['forecast'])" in content[auto_arima_section.index(prev_line)-1]:
                            fixed_section.append(" " * (auto_arima_indent + 24) + stripped_line)
                        else:
                            fixed_section.append(" " * (auto_arima_indent + 20) + stripped_line)
                    else:
                        fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "st.error" in stripped_line and "Error in Auto ARIMA model:" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                elif "if future_index is not None:" in stripped_line:
                    fixed_section.append(" " * (auto_arima_indent + 16) + stripped_line)
                else:
                    # Default indentation for lines inside the spinner block
                    fixed_section.append(" " * (auto_arima_indent + 12) + stripped_line)
        
        # Replace the original section with the fixed one
        content[start_line:end_line] = fixed_section
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.writelines(content)
        
        return True
    
    return False

if __name__ == "__main__":
    success = fix_indentation()
    print(f"Indentation fix {'successful' if success else 'failed'}")
