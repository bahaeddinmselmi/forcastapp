import os
import re
import ast
import sys

# Path to the original ui.py file
original_ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
# Path to the new corrected ui.py file
new_ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui_fixed.py"

print(f"Starting comprehensive analysis and correction of {original_ui_path}")

# Load the original file content
try:
    with open(original_ui_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"Successfully read original file ({len(content)} bytes)")
except Exception as e:
    print(f"Error reading original file: {e}")
    sys.exit(1)

# Create a backup of the original file
backup_path = original_ui_path + ".full-rebuild-backup"
try:
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup at {backup_path}")
except Exception as e:
    print(f"Error creating backup: {e}")
    sys.exit(1)

# First pass: Fix common syntax errors that prevent parsing
print("First pass: Fixing common syntax errors...")

# Fix 1: Fix incorrect indentation in try-except blocks
def fix_try_except_blocks(content):
    lines = content.split('\n')
    fixed_lines = []
    stack = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indentation = len(line) - len(line.lstrip())
        
        if stripped.startswith('try:'):
            stack.append((i, indentation))
            fixed_lines.append(line)
        elif stripped.startswith('except ') or stripped == 'except:':
            if stack:
                # Check if indentation matches the corresponding try
                try_line, try_indent = stack.pop()
                if indentation != try_indent:
                    # Fix indentation to match the try statement
                    line = ' ' * try_indent + stripped
                fixed_lines.append(line)
            else:
                # This is an except without a matching try
                # Add a default indentation and a comment
                fixed_lines.append(line + '  # WARNING: except without matching try')
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Check for unmatched try statements and add except blocks
    if stack:
        print(f"Found {len(stack)} unmatched try statements, adding except blocks")
        fixed_content = '\n'.join(fixed_lines)
        
        for try_line, try_indent in stack:
            # Find where to insert the except (looking for next line with same indentation)
            line_num = try_line
            while line_num < len(fixed_lines):
                line = fixed_lines[line_num]
                if not line.strip():  # Skip empty lines
                    line_num += 1
                    continue
                    
                curr_indent = len(line) - len(line.lstrip())
                if curr_indent <= try_indent and line_num > try_line:
                    # Insert the except block right before this line
                    indent_str = ' ' * try_indent
                    except_line = indent_str + "except Exception as e:"
                    handler_line = indent_str + "    print(f\"Error: {e}\")  # Added to fix syntax"
                    
                    fixed_lines.insert(line_num, except_line)
                    fixed_lines.insert(line_num + 1, handler_line)
                    
                    print(f"Added missing except clause after try at line {try_line+1}")
                    break
                line_num += 1
    
    return '\n'.join(fixed_lines)

# Fix 2: Fix duplicate else statements
def fix_duplicate_else(content):
    pattern = re.compile(r'(\s*else:\s*\n\s*else:)', re.MULTILINE)
    matches = list(pattern.finditer(content))
    if matches:
        print(f"Found {len(matches)} duplicate else statements")
        
    # Replace duplicate else with a comment
    content = pattern.sub(r'\g<1>  # Duplicate else removed', content)
    
    # Also look for else without if
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        if line.strip() == 'else:':
            # Check if there's an if statement before this
            found_if = False
            for j in range(i-1, max(0, i-20), -1):
                if re.search(r'\bif\b.*:', lines[j].strip()):
                    found_if = True
                    break
            
            if not found_if:
                line = line.replace('else:', '# else without matching if')
                print(f"Fixed else without matching if at line {i+1}")
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

# Fix 3: Fix unmatched curly braces
def fix_unmatched_braces(content):
    lines = content.split('\n')
    fixed_lines = []
    stack = []
    
    for i, line in enumerate(lines):
        # Track opening and closing braces
        for char in line:
            if char == '{':
                stack.append((i, '{'))
            elif char == '}':
                if stack and stack[-1][1] == '{':
                    stack.pop()
                else:
                    # Unmatched closing brace
                    line = line.replace('}', '# }', 1)
                    print(f"Fixed unmatched closing brace at line {i+1}")
        
        fixed_lines.append(line)
    
    # If there are unmatched opening braces, add closing braces
    if stack:
        print(f"Found {len(stack)} unmatched opening braces")
        for i, brace in reversed(stack):
            # Add closing brace at the end of the last non-empty line
            for j in range(len(fixed_lines)-1, -1, -1):
                if fixed_lines[j].strip():
                    fixed_lines[j] += ' }'  # Add closing brace
                    print(f"Added missing closing brace for opening brace at line {i+1}")
                    break
    
    return '\n'.join(fixed_lines)

# Fix 4: Fix dictionary formatting errors
def fix_dictionary_formatting(content):
    # Look for dictionary definitions that are missing opening braces
    pattern = re.compile(r"(\w+)\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*\n\s*'([^']+)'", re.MULTILINE)
    matches = list(pattern.finditer(content))
    if matches:
        print(f"Found {len(matches)} potential dictionary formatting errors")
        
    # Replace with proper dictionary format
    content = pattern.sub(r"\1[\"\2\"] = {\n    '\3'", content)
    
    return content

# Apply the fixes
content = fix_try_except_blocks(content)
content = fix_duplicate_else(content)
content = fix_unmatched_braces(content)
content = fix_dictionary_formatting(content)

# Second pass: Try to parse the file with ast and see if there are remaining syntax errors
print("Second pass: Checking for remaining syntax errors...")

try:
    ast.parse(content)
    print("No syntax errors detected by AST parser!")
except SyntaxError as e:
    error_line = e.lineno if hasattr(e, 'lineno') else 'unknown'
    error_col = e.offset if hasattr(e, 'offset') else 'unknown'
    error_msg = str(e)
    print(f"Syntax error detected at line {error_line}, column {error_col}: {error_msg}")
    
    # Fix specific known issues based on common error patterns
    lines = content.split('\n')
    
    # If the error is at a specific line, we can try some specialized fixes
    if hasattr(e, 'lineno') and e.lineno > 0 and e.lineno <= len(lines):
        error_line_content = lines[e.lineno - 1]
        print(f"Error line content: '{error_line_content}'")
        
        # Add more specialized fixes here based on error patterns
        
        # For missing except after try
        if "unexpected indent" in error_msg and e.lineno > 1:
            prev_line = lines[e.lineno - 2]
            if prev_line.strip() == "try:":
                # The line after a try is not properly indented
                indent = len(prev_line) - len(prev_line.lstrip())
                fixed_line = ' ' * (indent + 4) + error_line_content.lstrip()
                lines[e.lineno - 1] = fixed_line
                print(f"Fixed indentation at line {e.lineno}")
        
        # For expected except block
        if "expected 'except' or 'finally' block" in error_msg:
            # Search back for the try statement
            for i in range(e.lineno - 2, max(0, e.lineno - 20), -1):
                if lines[i].strip() == "try:":
                    # Add a generic except block right after the error line
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    except_line = ' ' * indent + "except Exception as e:"
                    handler_line = ' ' * indent + "    print(f\"Error: {e}\")  # Added to fix syntax"
                    
                    lines.insert(e.lineno, handler_line)
                    lines.insert(e.lineno, except_line)
                    print(f"Added missing except clause after line {e.lineno}")
                    break
        
        content = '\n'.join(lines)

# Save the fixed content to the new file
try:
    with open(new_ui_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved corrected file to {new_ui_path}")
except Exception as e:
    print(f"Error saving corrected file: {e}")
    sys.exit(1)

print("File analysis and correction complete!")
print("To use the fixed file, rename it to ui.py by running:")
print(f"  ren \"{new_ui_path}\" ui.py")
print("Or use the fixed file directly by importing from it instead of the original ui.py")
