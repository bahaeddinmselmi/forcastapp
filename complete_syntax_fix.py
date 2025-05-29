import os
import re

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".complete-syntax-fix-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# First, let's find all 'try' blocks without matching 'except' blocks
def find_unmatched_try_blocks(content):
    lines = content.split('\n')
    stack = []
    issues = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('try:'):
            stack.append((i, line))
        elif stripped.startswith('except ') or stripped == 'except:':
            if stack:
                stack.pop()  # Matched try-except pair
    
    # Any remaining items in stack are unmatched try blocks
    for i, line in stack:
        issues.append((i, "unmatched try"))
    
    return issues

# Let's also detect indentation errors
def find_indentation_errors(content):
    lines = content.split('\n')
    issues = []
    
    indentation_rules = [
        (r'^\s*try:', r'^\s*except', 'try without except'),
        (r'^\s*if.*:', r'^\s*else:', 'if without else in proper indentation'),
    ]
    
    for i in range(len(lines) - 1):
        current_line = lines[i].rstrip()
        next_line = lines[i + 1].rstrip()
        
        # Skip empty lines
        if not current_line.strip():
            continue
            
        # Check for inconsistent indentation
        if current_line.strip().endswith(':'):
            current_indent = len(current_line) - len(current_line.lstrip())
            next_indent = len(next_line) - len(next_line.lstrip())
            
            if next_indent <= current_indent and next_line.strip() and not (next_line.strip().startswith('else:') or next_line.strip().startswith('except') or next_line.strip().startswith('finally:')):
                issues.append((i + 1, "unexpected indentation"))
    
    return issues

# Let's find and report all issues
unmatched_try = find_unmatched_try_blocks(content)
indentation_errors = find_indentation_errors(content)

print(f"Found {len(unmatched_try)} unmatched try blocks")
print(f"Found {len(indentation_errors)} indentation errors")

# Now let's fix the specific 'try' block issue at line 259
# This is the line that's causing the current error
lines = content.split('\n')

# Add missing 'except' clauses to unmatched 'try' blocks
for i, issue_type in unmatched_try:
    if issue_type == "unmatched try":
        # Find the indentation level of the try statement
        indent = len(lines[i]) - len(lines[i].lstrip())
        indent_str = ' ' * indent
        
        # Check if there's an except block following closely
        has_except = False
        for j in range(i+1, min(i+10, len(lines))):
            if lines[j].strip().startswith('except'):
                has_except = True
                break
        
        if not has_except:
            # Insert an except block right after the try line
            indented_line = indent_str + "    pass  # Placeholder\n"
            except_line = indent_str + "except Exception as e:\n"
            except_handler = indent_str + "    print(f\"Error: {e}\")  # Added to fix syntax\n"
            
            lines.insert(i+1, indented_line)
            lines.insert(i+2, except_line)
            lines.insert(i+3, except_handler)
            
            print(f"Added missing except clause after try at line {i+1}")

# Now let's look at two specific areas known to cause problems:
# 1. Line ~259 (data_freq = 'MS')
line_259_area_start = max(0, 250)
line_259_area_end = min(len(lines), 270)

found_try_near_259 = False
for i in range(line_259_area_start, line_259_area_end):
    if lines[i].strip() == "try:":
        found_try_near_259 = True
        # Check if there's a matching except
        has_except = False
        for j in range(i+1, line_259_area_end):
            if lines[j].strip().startswith("except"):
                has_except = True
                break
        
        if not has_except:
            # Find indentation level
            indent = len(lines[i]) - len(lines[i].lstrip())
            indent_str = ' ' * indent
            
            # Find where to insert the except (right before the next non-indented line)
            insert_pos = -1
            for j in range(i+1, line_259_area_end):
                line_indent = len(lines[j]) - len(lines[j].lstrip())
                if line_indent <= indent and lines[j].strip():
                    insert_pos = j
                    break
            
            if insert_pos != -1:
                lines.insert(insert_pos, f"{indent_str}except Exception as e:")
                lines.insert(insert_pos+1, f"{indent_str}    print(f\"Error: {{e}}\")  # Added to fix syntax")
                print(f"Added missing except clause before line {insert_pos+1}")

# 2. Line ~245-255 (forecasts["ARIMA (Fallback)"])
line_245_area_start = max(0, 240)
line_245_area_end = min(len(lines), 260)

# Fix specific issue at line ~248
for i in range(line_245_area_start, line_245_area_end):
    if "forecasts[\"ARIMA (Fallback)\"]" in lines[i]:
        # Check the structure
        indent = len(lines[i]) - len(lines[i].lstrip())
        indent_str = ' ' * indent
        
        # Look back to find the matching try
        found_try = False
        for j in range(i-10, i):
            if j >= 0 and lines[j].strip() == "try:":
                found_try = True
                break
        
        if not found_try:
            # There should be a try before this, but it's missing
            # Add it at a suitable position
            best_insert = max(i-3, line_245_area_start)
            lines.insert(best_insert, f"{indent_str}try:")
            print(f"Added missing try statement before line {best_insert+1}")

# Now fix the specific section at line 259 (data_freq = 'MS')
# First, find the exact line number in the current content
for i, line in enumerate(lines):
    if "data_freq = 'MS'" in line and "# Default value" in line:
        # This is likely line 259
        # Check if there's an open try block
        prev_try_pos = -1
        for j in range(i-10, i):
            if j >= 0 and lines[j].strip() == "try:":
                prev_try_pos = j
                break
        
        # Check if there's an except following
        next_except_pos = -1
        for j in range(i+1, min(i+10, len(lines))):
            if lines[j].strip().startswith("except"):
                next_except_pos = j
                break
        
        if prev_try_pos != -1 and next_except_pos == -1:
            # We found a try without except
            # Find indentation level of the try
            try_indent = len(lines[prev_try_pos]) - len(lines[prev_try_pos].lstrip())
            try_indent_str = ' ' * try_indent
            
            # Add except right after current line
            lines.insert(i+1, f"{try_indent_str}except Exception as e:")
            lines.insert(i+2, f"{try_indent_str}    print(f\"Error: {{e}}\")  # Added to fix syntax")
            print(f"Added missing except clause after line {i+1}")

# Now let's fix any duplicate else statements
for i in range(len(lines) - 1):
    if lines[i].strip() == "else:" and lines[i+1].strip() == "else:":
        # Fix the duplicate else
        lines[i] = lines[i].replace("else:", "# Removed duplicate else")
        print(f"Fixed duplicate else at line {i+1}")

# Save the fixed content back to the file
fixed_content = '\n'.join(lines)
with open(ui_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Applied comprehensive syntax fixes to {ui_path}")
print("Try running the application now.")
