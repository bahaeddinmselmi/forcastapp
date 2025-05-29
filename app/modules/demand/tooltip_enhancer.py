"""
Tooltip enhancement for plotly figures in the IBP system
"""

def enhance_tooltips(fig):
    """
    Enhance a Plotly figure's tooltips to show exact values
    
    Args:
        fig: The Plotly figure to enhance
        
    Returns:
        The enhanced figure
    """
    # Add detailed tooltip styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Configure hover template to show exact values
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'hovertemplate') and trace.hovertemplate is not None:
            fig.data[i].hovertemplate = '%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>'
    
    return fig
