import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Forecasting Test App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("📊 Forecasting Test Application")
st.markdown("""
This is a test version of the forecasting application. 
It demonstrates basic functionality with sample data.
""")

# Create sample data
@st.cache_data
def generate_sample_data():
    dates = pd.date_range(start='2024-01-01', periods=24, freq='M')
    np.random.seed(42)
    values = np.random.normal(100, 20, 24) + np.sin(np.arange(24)/2)*10 + np.arange(24)*2
    df = pd.DataFrame({
        'Date': dates,
        'Sales': values
    })
    return df

# Generate and display sample data
data = generate_sample_data()

# Sidebar controls
st.sidebar.title("Test Controls")
show_data = st.sidebar.checkbox("Show Raw Data", value=True)
chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Bar"])

# Display data if checkbox is selected
if show_data:
    st.subheader("Sample Data")
    st.dataframe(data)

# Create and display chart
st.subheader("Sample Visualization")
if chart_type == "Line":
    fig = px.line(data, x='Date', y='Sales', title='Sample Sales Trend')
else:
    fig = px.bar(data, x='Date', y='Sales', title='Sample Sales by Month')

st.plotly_chart(fig, use_container_width=True)

# Add some metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Sales", f"{data['Sales'].mean():.1f}")
with col2:
    st.metric("Maximum Sales", f"{data['Sales'].max():.1f}")
with col3:
    st.metric("Minimum Sales", f"{data['Sales'].min():.1f}")

st.success("✅ Test app is running successfully!")
