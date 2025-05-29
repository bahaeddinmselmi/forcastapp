# How to Use Excel Files with the IBP System

If you're experiencing issues uploading Excel files through the browser interface, you can now use our new directory-based file loading feature.

## NEW FEATURE: Configure Year for Month-Only Data

If your Excel files only contain month names (like "January", "February") without specifying years, you can now manually set which year these months belong to:

1. After loading your Excel file, look for the **"CONFIGURE YEAR FOR MONTH-ONLY DATA"** section
2. Select your desired year (default is the current year)
3. Click **"APPLY SELECTED YEAR"** button
4. The system will automatically detect columns with month names and apply the selected year

This feature is available both when uploading files through the browser and when loading from directory.

## General Instructions:

1. Place your Excel files in this directory:
   ```
   C:\Users\Public\Downloads\ibp\dd\app\data\sample_files\
   ```

2. Rename your files based on which module you want to use:
   - For Demand Planning: `demand_data.xlsx`
   - For Inventory Optimization: `inventory_data.xlsx`
   - For S&OP: `sop_data.xlsx`

3. Launch the IBP application

4. In the data source selection, choose "Load From Directory" instead of "Upload Data"

## Troubleshooting

If you don't see your data:
- Make sure your Excel file is in the correct format
- Verify the file is saved in the exact path mentioned above
- Try using the "Manual Excel Import Options" if the AI detection doesn't correctly identify your data tables

## Example

To use your "application forcast.xlsx" file:

1. Copy it to the sample_files directory with the appropriate name:
   ```
   Copy "C:\path\to\application forcast.xlsx" "C:\Users\Public\Downloads\ibp\dd\app\data\sample_files\demand_data.xlsx"
   ```

2. Select "Load From Directory" in the Demand Planning module
