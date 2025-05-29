"""
This script directly fixes the syntax error in ui.py by locating all try blocks
and ensuring they have properly matched except blocks.
"""
import re

# Path to the ui.py file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py"

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Create a backup
with open(file_path + '.direct_backup', 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Created backup at {file_path}.direct_backup")

# Split into lines
lines = content.split('\n')

# Find the pattern of try blocks with matching except blocks
# We'll track indentation levels to identify mismatched blocks
stack = []
problem_try_lines = []

for i, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        continue  # Skip empty lines and comments
    
    indent = len(line) - len(line.lstrip())
    
    if stripped.startswith('try:'):
        # Push onto stack: [line_number, indent_level]
        stack.append([i, indent])
    
    elif stripped.startswith('except ') or stripped.startswith('finally:'):
        # If we have a matching try, pop it
        if stack and stack[-1][1] == indent:
            stack.pop()
        else:
            print(f"Warning: except/finally at line {i+1} doesn't match any try block")
    
    # If we're starting a new block with same indentation as an unclosed try,
    # this could indicate a missing except block (syntax error)
    elif stack and indent <= stack[-1][1] and not stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ')):
        problem_try_lines.append(stack[-1][0])
        stack.pop()

# Also check for any remaining unclosed try blocks at end of file
for try_line, indent in stack:
    problem_try_lines.append(try_line)

# Fix each problem
problem_try_lines = sorted(set(problem_try_lines))
print(f"Found {len(problem_try_lines)} try blocks without matching except/finally")

# We'll insert except blocks after the try blocks
# Start from the last one to avoid affecting line numbers
fixed_lines = lines.copy()
for try_line in reversed(problem_try_lines):
    try_indent = len(lines[try_line]) - len(lines[try_line].lstrip())
    
    # Find where to insert the except
    insert_line = try_line + 1
    while insert_line < len(lines):
        line = lines[insert_line]
        if not line.strip() or line.strip().startswith('#'):
            insert_line += 1
            continue
        
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= try_indent:
            break
        insert_line += 1
    
    # Create the except block with proper indentation
    except_block = [
        ' ' * try_indent + 'except Exception as e:',
        ' ' * (try_indent + 4) + 'st.warning(f"Error: {str(e)}")'
    ]
    
    # Insert the except block
    fixed_lines = fixed_lines[:insert_line] + except_block + fixed_lines[insert_line:]
    
    print(f"Fixed try block at line {try_line+1} by adding except at line {insert_line+1}")

# Write the fixed content
with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print(f"Fixed the file at {file_path}")
print("You can now try running the application again.")
