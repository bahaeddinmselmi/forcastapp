"""Script to fix indentation in ui.py"""

import re

# Path to the ui.py file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py"

# Create a backup of the original file
with open(file_path, 'r', encoding='utf-8') as f:
    original_content = f.read()

backup_path = file_path + '.bak'
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(original_content)

print(f"Created backup at {backup_path}")

# This function fixes the specific issue with try blocks without except blocks
def fix_try_blocks(content):
    # Find the specific problem try block at around line 2698
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    in_try_block = False
    needs_except = False
    indent_level = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Detect start of a try block
        if stripped.startswith('try:'):
            in_try_block = True
            needs_except = True
            indent_level = len(line) - len(line.lstrip())
        
        # Detect end of a try block with an except
        elif in_try_block and stripped.startswith('except'):
            needs_except = False
            in_try_block = False
        
        # If we're at a line that suggests the try block ended without an except
        elif in_try_block and not stripped.startswith('#') and not line.isspace() and not stripped == '':
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and "# Generate forecasts for the selected models" in line:
                # Add the missing except block
                except_line = ' ' * indent_level + 'except Exception as e:'
                warning_line = ' ' * (indent_level + 4) + 'st.warning(f"Error: {str(e)}")'                
                fixed_lines.append(except_line)
                fixed_lines.append(warning_line)
                fixed_lines.append('')
                in_try_block = False
                needs_except = False
        
        fixed_lines.append(line)
        i += 1
    
    # Make sure there's no open try block at the end
    if needs_except:
        indent_level = 20  # Common indentation level
        except_line = ' ' * indent_level + 'except Exception as e:'
        warning_line = ' ' * (indent_level + 4) + 'st.warning(f"Error: {str(e)}")'      
        fixed_lines.append(except_line)
        fixed_lines.append(warning_line)
    
    return '\n'.join(fixed_lines)

# Fix the indentation issues
fixed_content = fix_try_blocks(original_content)

# Save the fixed content
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Fixed indentation in {file_path}")
print("You can now run the app.")

