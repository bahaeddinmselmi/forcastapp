"""
Direct fix for the persistent syntax error in ui.py
"""
import shutil

def fix_ui_file():
    file_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    backup_path = file_path + ".final-fix-backup"
    
    # Make one final backup
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Locate the problematic section
    problem_section_start = """                                 st.error(f"Even fallback forecast failed: {fallback_error}")
                                 
         # If we have a DatetimeIndex, use our improved frequency detection
         # If we have a DatetimeIndex, use our improved frequency detection
         try:"""
    
    # Create the fixed section
    fixed_section = """                                 st.error(f"Even fallback forecast failed: {fallback_error}")
                                 
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
        """
    
    # Replace the problematic section with the fixed section
    if problem_section_start in content:
        # Get the end of the problematic section
        problem_section_end = "except Exception as e:\n"
        end_pos = content.find(problem_section_end, content.find(problem_section_start))
        if end_pos > 0:
            end_pos += len(problem_section_end)
            # Calculate where to continue from (next line after the except)
            next_line_pos = content.find('\n', end_pos) + 1
            
            # Replace the section
            fixed_content = content[:content.find(problem_section_start)] + fixed_section + content[next_line_pos:]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
                
            print("Fixed the problematic try-except section")
        else:
            print("Could not locate the end of the problematic section")
    else:
        # If we can't find the exact section, let's try a more direct approach
        # This uses line-by-line replacement for the specific problematic area
        print("Using alternate approach...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Find any line with "try:" and without a matching "except" within a reasonable distance
        for i in range(len(lines) - 1, -1, -1):  # Start from the end to avoid confusion
            if "try:" in lines[i]:
                # Check if there's an except within 15 lines
                found_except = False
                for j in range(i + 1, min(i + 15, len(lines))):
                    if "except " in lines[j]:
                        found_except = True
                        break
                
                if not found_except:
                    # Add an except block after this try
                    # First, get the indentation level
                    indentation = lines[i][:lines[i].find("try:")]
                    
                    # Look for the end of the try block - a line with same indentation level
                    try_block_end = i
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and lines[j].startswith(indentation) and not lines[j][len(indentation):].startswith(" "):
                            try_block_end = j
                            break
                    
                    # Insert except block
                    lines.insert(try_block_end, f"{indentation}except Exception as e:\n")
                    lines.insert(try_block_end + 1, f"{indentation}    print(f\"Error in try block: {{str(e)}}\")\n")
                    
                    print(f"Added missing except block after line {i+1}")
                    
                    # Ensure we don't find any more from this line upward
                    i = try_block_end + 2
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    print("Fix completed. Try running your Streamlit app now.")

if __name__ == "__main__":
    fix_ui_file()
