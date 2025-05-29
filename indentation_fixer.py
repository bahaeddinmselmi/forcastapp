"""
Simple script to fix indentation issues by adding proper except blocks
where they're missing in the ui.py file.
"""

def fix_indentation_issues():
    file_path = "C:/Users/Public/Downloads/ibp/dd/app/modules/demand/ui.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create a backup
    with open(file_path + '.bak', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    # Simple fix - replace problematic section around line 2690-2705
    # This approach replaces the section with properly indented code
    fixed_lines = []
    in_problematic_section = False
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Identify the problematic section
        if 2690 <= line_num <= 2710:
            if line_num == 2690:
                in_problematic_section = True
                # Add properly indented code
                fixed_lines.append(line)
            elif line_num == 2710:
                in_problematic_section = False
                fixed_lines.append(line)
            elif line_num == 2694:
                # Fix the indentation for the comment line
                fixed_lines.append("        # If we have a DatetimeIndex, use our improved frequency detection\n")
            elif line_num == 2696:
                # Ensure try block is correctly indented
                fixed_lines.append("        try:\n")
            elif line_num == 2704:
                # Ensure except block is correctly indented
                fixed_lines.append("        except Exception as e:\n")
            else:
                # Keep the line but fix indentation if it's part of the try/except
                if 2697 <= line_num <= 2703:
                    # Indentation for the try block content
                    if line.strip():
                        fixed_lines.append("            " + line.lstrip())
                    else:
                        fixed_lines.append(line)
                elif 2705 <= line_num <= 2709:
                    # Indentation for the code after try/except
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation issues in {file_path}")

if __name__ == "__main__":
    fix_indentation_issues()
