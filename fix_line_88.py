import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".line88-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the file line by line
with open(ui_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the indentation at line 88
lines[87] = "    else:\n"  # Fix the indentation of the else statement to match the if statement

# Save the modified file
with open(ui_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Fixed indentation at line 88 in {ui_path}")
print("Try running the application now.")
