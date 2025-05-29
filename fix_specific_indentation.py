"""Fix the specific indentation error in ui.py"""

# Path to the file
file_path = "C:\\Users\\Public\\Downloads\\ibp\\dd\\app\\modules\\demand\\ui.py"

# Create a backup of the file
backup_path = file_path + ".bak2"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created backup at {backup_path}")

# Now fix the specific problem with try without except at line ~2698
lines = content.split('\n')

# Look for the try block that's missing its except
try_line_numbers = []
try_block_levels = []
except_line_numbers = []

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith('try:'):
        indent_level = len(line) - len(line.lstrip())
        try_line_numbers.append(i)
        try_block_levels.append(indent_level)
    elif stripped.startswith('except '):
        except_line_numbers.append(i)

# Find which try block doesn't have a matching except
# Map try to except
try_except_map = {}
for i, try_line in enumerate(try_line_numbers):
    # Find the closest except after this try line
    matching_excepts = [e for e in except_line_numbers if e > try_line]
    if matching_excepts:
        try_except_map[try_line] = min(matching_excepts)
    else:
        try_except_map[try_line] = None

# Find try lines without matching except (those are our targets)
for try_line, except_line in try_except_map.items():
    if except_line is None:
        print(f"Found try without except at line {try_line+1}: {lines[try_line]}")
        # Now find where to insert the except
        indent_level = try_block_levels[try_line_numbers.index(try_line)]
        
        # Find the end of the try block (where indentation returns to the same level or less)
        try_end = None
        for i in range(try_line + 1, len(lines)):
            if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
                try_end = i
                break
        
        if try_end is not None:
            # Insert the missing except block before the line where indentation returns to try level
            indent_str = ' ' * indent_level
            missing_except = f"{indent_str}except Exception as e:\n{indent_str}    st.warning(f\"Error: {{str(e)}}\")"
            lines.insert(try_end, missing_except)
            print(f"Inserted missing except block at line {try_end+1}")

# Save the fixed content
fixed_content = '\n'.join(lines)
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Fixed indentation issues in {file_path}")
print("You can now run the app.")
