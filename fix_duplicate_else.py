import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".duplicate-else-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the duplicate else statements at lines 87-88
# First, check what's actually in those lines
print(f"Line 87: {lines[86]}")
print(f"Line 88: {lines[87]}")

# Fix the double else by removing one of them
if "else:" in lines[86] and "else:" in lines[87]:
    lines[87] = "        # Calculate a simple linear trend\n"
    print("Removed duplicate else statement at line 88")

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed duplicate else statements in {ui_path}")
print("Try running the application now.")
