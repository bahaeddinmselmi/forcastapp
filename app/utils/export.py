"""
Export utilities for the IBP system.
Provides functions for exporting data to various formats.
"""

import pandas as pd
import streamlit as st
import io
import base64
from typing import List, Dict, Any, Optional, Union
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

def create_excel_download_link(df: pd.DataFrame, title: str, sheet_name: str = "Data") -> str:
    """
    Generate a download link for a DataFrame as an Excel file.
    
    Args:
        df: DataFrame to export
        title: Title to display on the download button
        sheet_name: Name of the sheet in the Excel file
        
    Returns:
        HTML string with download link
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Get the xlsxwriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # Add some formatting
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#0066B7',  # Blue header
        'font_color': 'white',
        'border': 1
    })
    
    # Apply formatting to header row
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        # Set column width based on content
        max_len = max(df[value].astype(str).str.len().max(), len(str(value)) + 2)
        worksheet.set_column(col_num, col_num, max_len)
    
    # Add alternating row colors
    for row_num in range(1, len(df) + 1):
        if row_num % 2 == 0:
            # Even rows - light blue
            row_format = workbook.add_format({'bg_color': '#E6F0FF'})
            worksheet.set_row(row_num, None, row_format)
    
    writer.close()
    
    # Generate download link
    b64 = base64.b64encode(output.getvalue()).decode()
    filename = f"{title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.xlsx"
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-link">Download {title} as Excel file</a>'
    return href

def generate_formatted_report(df: pd.DataFrame, title: str, fig: Optional[Union[go.Figure, plt.Figure]] = None, 
                             metadata: Optional[Dict[str, Any]] = None, sheet_name: str = "Data") -> io.BytesIO:
    """
    Generate a formatted Excel report with data, charts and metadata.
    
    Args:
        df: DataFrame with the main data
        title: Report title
        fig: Optional Plotly or Matplotlib figure to include
        metadata: Optional dictionary of metadata to include
        sheet_name: Name for the data sheet
        
    Returns:
        BytesIO object containing the Excel file
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Write data to main sheet
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # Add report title
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 16,
        'align': 'center',
        'valign': 'vcenter',
        'font_color': '#0066B7'
    })
    worksheet.merge_range('A1:H1', title, title_format)
    
    # Format the header row
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#0066B7',
        'font_color': 'white',
        'border': 1
    })
    
    # Apply formatting to header row
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(1, col_num, value, header_format)
        max_len = max(df[value].astype(str).str.len().max(), len(str(value)) + 2)
        worksheet.set_column(col_num, col_num, max_len)
    
    # Add alternating row colors
    for row_num in range(2, len(df) + 2):
        if row_num % 2 == 0:
            row_format = workbook.add_format({'bg_color': '#E6F0FF'})
            worksheet.set_row(row_num, None, row_format)
    
    # Add metadata if provided
    if metadata:
        # Create a metadata sheet
        meta_sheet = workbook.add_worksheet("Report Info")
        
        # Format the metadata sheet
        meta_title_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'font_color': '#0066B7'
        })
        meta_sheet.write(0, 0, "Report Information", meta_title_format)
        
        # Add metadata
        meta_key_format = workbook.add_format({'bold': True})
        for i, (key, value) in enumerate(metadata.items(), start=2):
            meta_sheet.write(i, 0, key, meta_key_format)
            meta_sheet.write(i, 1, str(value))
    
    # Add chart if provided
    if fig:
        # Add a charts sheet
        chart_sheet = workbook.add_worksheet("Charts")
        
        # If it's a Plotly figure, save as image
        if isinstance(fig, go.Figure):
            img_bytes = fig.to_image(format="png")
            chart_sheet.insert_image('B2', '', {'image_data': img_bytes})
        # If it's a Matplotlib figure, save as image
        elif isinstance(fig, plt.Figure):
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format='png')
            imgdata.seek(0)
            chart_sheet.insert_image('B2', '', {'image_data': imgdata.getvalue()})
    
    writer.close()
    output.seek(0)
    return output

def create_full_report_download_link(df: pd.DataFrame, title: str, fig: Optional[Union[go.Figure, plt.Figure]] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a download link for a complete formatted report.
    
    Args:
        df: DataFrame with the main data
        title: Report title
        fig: Optional Plotly or Matplotlib figure to include
        metadata: Optional dictionary of metadata to include
        
    Returns:
        HTML string with download link
    """
    # Generate the report
    output = generate_formatted_report(df, title, fig, metadata)
    
    # Create download link
    b64 = base64.b64encode(output.getvalue()).decode()
    filename = f"{title.replace(' ', '_').lower()}_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-link">Download Complete Report as Excel</a>'
    return href
