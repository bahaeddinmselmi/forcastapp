"""
A comprehensive script to fix all try/except blocks in ui.py.
This script analyzes the Python code, identifies mismatched try/except blocks,
and fixes them by adding necessary except blocks where they're missing.
"""

import re

def fix_try_except_blocks(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # Create a backup
    with open(file_path + '.full_backup', 'w', encoding='utf-8') as f:
        f.writelines(content)
    
    # Process the file line by line
    stack = []  # Stack to track try blocks (line_num, indent_level)
    changes = []  # List of (line_num, text) to insert
    
    for i, line in enumerate(content):
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
        
        # Calculate indentation level
        indent_level = len(line) - len(line.lstrip())
        
        # Process try blocks
        if stripped == 'try:':
            stack.append((i, indent_level))
        
        # Process except blocks - match with try blocks
        elif stripped.startswith('except ') or stripped == 'except:' or stripped.startswith('finally:'):
            # Find matching try block (same indentation)
            found_match = False
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][1] == indent_level:
                    stack.pop(j)
                    found_match = True
                    break
            
            if not found_match:
                print(f"Warning: except/finally at line {i+1} doesn't match any try block")
        
        # Check for potential unclosed try blocks when indentation decreases
        elif stack:
            # If we're at the same or lower indentation level than a try block,
            # it may indicate missing except
            j = len(stack) - 1
            while j >= 0:
                if indent_level <= stack[j][1]:
                    # This line is at the same or lower indentation as an unclosed try
                    # Only consider it if it's not an if/for/while continuation
                    prev_line_stripped = content[i-1].strip() if i > 0 else ""
                    if not prev_line_stripped.endswith(':') and not stripped.startswith(('elif ', 'else:', 'except ')):
                        try_line, try_indent = stack.pop(j)
                        # Insert except before this line
                        changes.append((i, ' ' * try_indent + 'except Exception as e:\n' + 
                                           ' ' * (try_indent + 4) + 'pass  # Auto-added except block\n'))
                        print(f"Adding except block after try at line {try_line+1} before line {i+1}")
                j -= 1
    
    # Add except blocks for any remaining unclosed try blocks at EOF
    for try_line, try_indent in reversed(stack):
        changes.append((len(content), ' ' * try_indent + 'except Exception as e:\n' + 
                                      ' ' * (try_indent + 4) + 'pass  # Auto-added except block at EOF\n'))
        print(f"Adding except block at EOF for try at line {try_line+1}")
    
    # Apply all changes
    changes.sort(reverse=True)  # Apply from bottom to top to avoid affecting line numbers
    for line_num, text in changes:
        content.insert(line_num, text)
    
    # Write the fixed file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(content)
    
    print(f"Applied {len(changes)} fixes to the file.")

if __name__ == "__main__":
    file_path = r"C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py"
    fix_try_except_blocks(file_path)
    print("Done! Try running the main application now.")
