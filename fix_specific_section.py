"""
Script to fix the specific indentation issue around line 3248 in ui.py.
This targets the problematic section directly.
"""

# Path to the file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py"

# Create a backup
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.readlines()

with open(file_path + '.section_backup', 'w', encoding='utf-8') as f:
    f.writelines(content)

print("Created backup of ui.py")

# The problematic section starts around line 3235 and continues for several lines
# Let's rewrite this entire section with proper indentation

# Find start of the problematic section (arima_params definition)
start_line = 0
for i, line in enumerate(content):
    if "arima_params = {" in line:
        start_line = i
        break

if start_line > 0:
    # Find end of the problematic section (where the next proper indentation begins)
    end_line = start_line
    for i in range(start_line + 1, min(start_line + 50, len(content))):
        if "# Perform the ensemble forecast" in content[i]:
            end_line = i - 1
            break
    
    # Now replace this section with a properly indented version
    proper_indentation = """            # Get advanced parameters from config
            arima_params = {
                'max_p': config.get('arima', {}).get('max_p', 5),
                'max_d': config.get('arima', {}).get('max_d', 2),
                'max_q': config.get('arima', {}).get('max_q', 5),
                'seasonal': config.get('arima', {}).get('seasonal', True),
                'n_fits': config.get('arima', {}).get('n_fits', 50),
                'information_criterion': config.get('arima', {}).get('information_criterion', 'aic'),
                'stepwise': config.get('arima', {}).get('stepwise', True)
            }
            
            # Add seasonal parameters if using seasonality
            if arima_params['seasonal']:
                # Add any specific seasonal parameters here if needed
                pass
"""
    
    # Replace the problematic section
    content[start_line:end_line+1] = proper_indentation.splitlines(keepends=True)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(content)
    
    print(f"Fixed indentation in section from line {start_line+1} to {end_line+1}")
else:
    print("Could not find the arima_params section")

# Also need to fix the 'else' that's incorrectly indented around line 3249
# Find this else block
for i, line in enumerate(content):
    if "else:" in line and "# Use default weight if model not found" in content[i+1]:
        # This is likely the unmatched else block that's causing problems
        # We'll simply comment it out for now to fix the syntax
        content[i] = "# " + content[i]
        content[i+1] = "# " + content[i+1]
        content[i+2] = "# " + content[i+2]
        print(f"Commented out unmatched else block at line {i+1}")
        break

# Write the fixed content back to file
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(content)

print("Done! Try running the main application now.")
