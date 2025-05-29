"""
Script to systematically fix all try/except blocks in ui.py
"""
import os
import re

def fix_try_except_blocks():
    ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    backup_path = ui_path + ".final-backup"
    
    # Create backup
    if not os.path.exists(backup_path):
        with open(ui_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Created backup at {backup_path}")
    
    # Approach: Let's add the key except block at line ~2697 by checking
    # all try statements and ensuring they have matching except blocks
    try:
        with open(ui_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add a specific except block for the try at line ~2697 if it's missing one
        pattern = r"try:\s+if isinstance\(train_data\.index, pd\.DatetimeIndex\):"
        matches = list(re.finditer(pattern, content))
        
        for match in matches:
            # Check if this try has a matching except within the next 20 lines
            match_pos = match.start()
            next_20_lines = content[match_pos:match_pos + 1000]  # Look ahead enough
            
            if "except Exception as" not in next_20_lines[:next_20_lines.find("try:") if "try:" in next_20_lines else len(next_20_lines)]:
                # No matching except found - this is the culprit
                print(f"Found a try without matching except at position {match_pos}")
                
                # Get the indentation level
                line_start = content.rfind('\n', 0, match_pos) + 1
                indentation = content[line_start:match_pos]
                
                # Find where to insert the except block
                # Look for the end of the try block - typically right before another try or a function
                end_markers = ["try:", "def ", "if __name__", "import ", "# "]
                end_pos = match_pos
                for marker in end_markers:
                    next_occurrence = content.find(marker, match_pos + 50)  # Skip the current try statement
                    if next_occurrence > 0 and (end_pos == match_pos or next_occurrence < end_pos):
                        end_pos = content.rfind('\n', match_pos, next_occurrence)
                
                if end_pos > match_pos:
                    # Insert except block before the end marker
                    insert_pos = end_pos
                    fixed_content = (
                        content[:insert_pos] + 
                        f"\n{indentation}except Exception as e:\n{indentation}    st.warning(f\"Error in frequency detection: {{str(e)}}\")\n{indentation}    data_freq = 'MS'\n" +
                        content[insert_pos:]
                    )
                    
                    # Write fixed content
                    with open(ui_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    print(f"Added missing except block at position {insert_pos}")
                else:
                    print("Could not determine where to insert the except block")
            else:
                print(f"Try block at {match_pos} already has a matching except")
    
    except Exception as e:
        print(f"Error fixing the file: {str(e)}")

if __name__ == "__main__":
    fix_try_except_blocks()
