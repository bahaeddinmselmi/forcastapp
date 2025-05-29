"""
Direct script to fix the try/except issue at line 2697
"""
import os

def fix_line_2697():
    file_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    backup_path = file_path + ".line2697-backup"
    
    # Create a backup
    if not os.path.exists(backup_path):
        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Created backup at {backup_path}")
    
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic try block around line 2697
    output_lines = []
    skip_until = -1
    fixed = False
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Skip lines if needed (already handled)
        if skip_until >= line_num:
            continue
            
        # Look for the problematic try block
        if line_num >= 2690 and line_num <= 2700 and "try:" in line and not fixed:
            # This is the problematic section - replace it with a corrected version
            indent = line[:line.find("try:")]
            output_lines.append(f"{indent}# If we have a DatetimeIndex, use our improved frequency detection\n")
            output_lines.append(f"{indent}try:\n")
            output_lines.append(f"{indent}    if isinstance(train_data.index, pd.DatetimeIndex):    \n")
            output_lines.append(f"{indent}        data_freq = get_data_frequency(train_data.index)\n")
            output_lines.append(f"{indent}        # Store it for future use\n")
            output_lines.append(f"{indent}        st.session_state['detected_frequency'] = data_freq\n")
            output_lines.append(f"{indent}    else:\n")
            output_lines.append(f"{indent}        # Default if all else fails\n")
            output_lines.append(f"{indent}        data_freq = 'MS'\n")
            output_lines.append(f"{indent}except Exception as e:\n")
            output_lines.append(f"{indent}    st.warning(f\"Error detecting frequency: {{str(e)}}. Using default frequency.\")\n")
            output_lines.append(f"{indent}    data_freq = 'MS'\n")
            output_lines.append(f"{indent}seasonal_order = None\n")
            output_lines.append(f"{indent}confidence_interval = 0.95\n")
            
            # Skip the next few lines which would have been the problematic section
            # Look for a logical end point to skip to
            for j in range(i+1, len(lines)):
                if "st.info(f\"Using frequency: {data_freq}" in lines[j]:
                    skip_until = j
                    fixed = True
                    break
            
            if not skip_until > i:
                # If we couldn't find a logical end point, skip a reasonable number of lines
                skip_until = i + 15
                fixed = True
        else:
            # Copy the line as is
            output_lines.append(line)
    
    # Write the fixed file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Fixed try/except block at line ~2697 in {file_path}")
    print("Run your Streamlit app to see if the issue is resolved")

if __name__ == "__main__":
    fix_line_2697()
