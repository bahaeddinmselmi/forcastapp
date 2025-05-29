"""Add warning suppression to the top of the forecasting module"""

# Warning suppression code to add
warning_code = """
# Add warning suppression
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence common warnings that affect forecasting
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")
"""

# Path to the file
file_path = "C:\\Users\\Public\\Downloads\\ibp\\dd\\app\\utils\\forecasting.py"

# Read the current content
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the docstring at the top of the file
docstring_end = content.find('"""', content.find('"""') + 3) + 3

# Insert the warning suppression code after the imports section
imports_end = content.find('\n\n', docstring_end)
if imports_end == -1:  # If there's no double newline after imports
    imports_end = content.find('\ndef', docstring_end)  # Find the first function definition

if imports_end != -1:
    # Insert the warning suppression code
    new_content = content[:imports_end] + warning_code + content[imports_end:]
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Successfully added warning suppression to {file_path}")
else:
    print("Could not find a suitable location to insert warning suppression code")
