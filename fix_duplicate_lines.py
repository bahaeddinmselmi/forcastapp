import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".duplicate-fix-backup"

# Create a backup
if not os.path.exists(backup_path):
    with open(ui_path, 'r', encoding='utf-8') as src:
        content = src.read()
    with open(backup_path, 'w', encoding='utf-8') as dst:
        dst.write(content)
    print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the duplicated lines around line 2695-2698
fixed_lines = []
skip_lines = False

for i, line in enumerate(lines):
    line_num = i + 1
    
    # Handle the duplicate code section
    if line_num == 2695:
        # This line has duplicate content that shouldn't be there
        fixed_lines.append("                                st.error(f\"Even fallback forecast failed: {fallback_error}\")\n")
        skip_lines = True
    elif skip_lines and line_num >= 2696 and line_num <= 2698:
        # Skip these lines as they're duplicated
        continue
    else:
        # Include all other lines normally
        fixed_lines.append(line)
        if line_num == 2699:
            skip_lines = False

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"Fixed duplicate code at lines 2695-2698 in {ui_path}")
print("Try running the application now.")
