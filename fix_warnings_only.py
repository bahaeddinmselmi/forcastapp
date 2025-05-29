"""Script to add warning suppression to forecasting.py without changing functions"""

# Path to the forecasting.py file
file_path = r"C:\Users\Public\Downloads\ibp\dd\app\utils\forecasting.py"

# Read the current content
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Warning suppression code to add
warning_imports = """
# Add warning suppression
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Silence common warnings that affect forecasting
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format")
warnings.filterwarnings("ignore", message="Covariance matrix is singular")
"""

# Find the position after the Prophet import
prophet_import_pos = content.find("from prophet import Prophet")
next_line_pos = content.find("\n", prophet_import_pos)

if prophet_import_pos != -1 and next_line_pos != -1:
    # Insert the warning suppression code
    modified_content = content[:next_line_pos+1] + warning_imports + content[next_line_pos+1:]
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("Added warning suppression to forecasting.py without changing functions")
else:
    print("Could not find the Prophet import line. No changes made.")
