import os

# Path to the ui.py file
ui_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
backup_path = ui_path + ".direct-fix-backup"

# Create a backup
with open(ui_path, 'r', encoding='utf-8') as src:
    content = src.read()
with open(backup_path, 'w', encoding='utf-8') as dst:
    dst.write(content)
print(f"Created backup at {backup_path}")

# Read the entire file content
with open(ui_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Look for the specific pattern with duplicate else statements
pattern = r"    if n_points <= 1:\n        # If only one point, use it as a constant forecast\n        last_value = train_data.iloc\[-1\]\n    else:\n        else:"
replacement = "    if n_points <= 1:\n        # If only one point, use it as a constant forecast\n        last_value = train_data.iloc[-1]\n        trend = 0\n    else:"

# Replace the pattern
if pattern in content:
    new_content = content.replace(pattern, replacement)
    print("Found and fixed duplicate else pattern")
    
    # Save the corrected file
    with open(ui_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Fixed duplicate else statements in {ui_path}")
else:
    print("Could not find the exact pattern with duplicate else statements")
    # Try a more flexible approach to finding the problem
    lines = content.split("\n")
    for i in range(len(lines)-1):
        if lines[i].strip() == "else:" and lines[i+1].strip() == "else:":
            lines[i] = lines[i].replace("else:", "# Removed duplicate else")
            print(f"Found and fixed duplicate else at line {i+1}")
            
            # Save the corrected file
            with open(ui_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            break

print("Try running the application now.")
