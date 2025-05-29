import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".indent-fix-2581-backup"

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

# Find and fix specific lines with indentation problems
for i in range(len(lines)):
    # Fix the indentation at line 2581-2583
    if i+1 == 2581:
        lines[i] = "                            else:\n"
    elif i+1 == 2582:
        lines[i] = "                                # Default if all else fails\n"
    elif i+1 == 2583:
        lines[i] = "                                data_freq = 'MS'\n"
    elif i+1 == 2584:
        # Fix indentation of the following lines as well
        lines[i] = "                            seasonal_order = None\n"
    elif i+1 == 2585:
        lines[i] = "                            confidence_interval = 0.95\n"

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed indentation at lines 2581-2585 in {ui_path}")
print("Try running the application now.")
