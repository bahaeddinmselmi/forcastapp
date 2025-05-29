"""
Enhanced Model Training Module for Demand Forecasting
This module provides advanced configuration options and training capabilities
for all forecasting models in the IBP system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_training')

def show_model_training_config():
    """
    Display and manage advanced training configuration options for all models
    
    Returns:
        Dictionary with the optimized configuration settings
    """
    st.subheader("⚙️ Advanced Model Training Configuration")
    
    # Create tabs for different model families
    config_tabs = st.tabs([
        "Auto ARIMA", 
        "Exponential Smoothing",
        "XGBoost",
        "LSTM",
        "Ensemble"
    ])
    
    # Initialize config dictionary
    config = {
        'advanced_features': True,
        'arima': {},
        'exp_smoothing': {},
        'xgboost': {},
        'lstm': {},
        'ensemble': {}
    }
    
    # ARIMA Configuration Tab
    with config_tabs[0]:
        st.markdown("### Auto ARIMA Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            config['arima']['seasonal'] = st.checkbox("Include Seasonality", value=True, key="arima_seasonal")
            config['arima']['max_p'] = st.slider("Max AR Order (p)", 1, 10, 5, key="arima_max_p")
            config['arima']['max_d'] = st.slider("Max Differencing (d)", 0, 3, 2, key="arima_max_d")
            config['arima']['max_q'] = st.slider("Max MA Order (q)", 1, 10, 5, key="arima_max_q")
            
        with col2:
            if config['arima']['seasonal']:
                config['arima']['seasonal_periods'] = st.selectbox(
                    "Seasonal Periods", 
                    options=[4, 7, 12, 24, 52, 365],
                    index=2,  # Default to 12 (monthly)
                    help="Number of time steps for a seasonal period",
                    key="arima_seasonal_periods"
                )
                config['arima']['max_P'] = st.slider("Max Seasonal AR (P)", 0, 3, 2, key="arima_max_P")
                config['arima']['max_D'] = st.slider("Max Seasonal Differencing (D)", 0, 2, 1, key="arima_max_D")
                config['arima']['max_Q'] = st.slider("Max Seasonal MA (Q)", 0, 3, 2, key="arima_max_Q")
            
        col3, col4 = st.columns(2)
        with col3:
            config['arima']['information_criterion'] = st.selectbox(
                "Information Criterion", 
                options=['aic', 'bic', 'hqic', 'oob'],
                index=0,
                help="Criterion for model selection",
                key="arima_info_criterion"
            )
            
            config['arima']['stepwise'] = st.checkbox(
                "Use Stepwise Search", 
                value=True,
                help="Faster but less thorough search",
                key="arima_stepwise"
            )
            
        with col4:
            config['arima']['n_fits'] = st.slider(
                "Number of ARIMA Fits", 
                10, 
                100, 
                50,
                help="More fits = more thorough but slower",
                key="arima_n_fits"
            )
            
            config['arima']['confidence_interval'] = st.slider(
                "Confidence Interval", 
                0.5, 
                0.99, 
                0.9, 
                step=0.01,
                help="For forecast bounds",
                key="arima_conf_interval"
            )
    
    # Exponential Smoothing Configuration Tab
    with config_tabs[1]:
        st.markdown("### Exponential Smoothing Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            config['exp_smoothing']['trend'] = st.selectbox(
                "Trend Type",
                options=[None, 'add', 'mul'],
                index=1,
                format_func=lambda x: "None" if x is None else ("Additive" if x == 'add' else "Multiplicative"),
                help="Type of trend component",
                key="es_trend_type"
            )
            
            if config['exp_smoothing']['trend'] is not None:
                config['exp_smoothing']['damped_trend'] = st.checkbox(
                    "Damped Trend", 
                    value=False,
                    help="Dampen the trend over future periods",
                    key="es_damped_trend"
                )
                
        with col2:
            config['exp_smoothing']['seasonal'] = st.selectbox(
                "Seasonal Type",
                options=[None, 'add', 'mul'],
                index=1,
                format_func=lambda x: "None" if x is None else ("Additive" if x == 'add' else "Multiplicative"),
                help="Type of seasonality component",
                key="es_seasonal_type"
            )
            
            if config['exp_smoothing']['seasonal'] is not None:
                config['exp_smoothing']['seasonal_periods'] = st.selectbox(
                    "Seasonal Periods", 
                    options=[4, 7, 12, 24, 52, 365],
                    index=2,  # Default to 12 (monthly)
                    help="Number of time steps for a seasonal period",
                    key="es_seasonal_periods"
                )
    
    # XGBoost Configuration Tab
    with config_tabs[2]:
        st.markdown("### XGBoost Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            config['xgboost']['max_depth'] = st.slider(
                "Max Tree Depth", 
                3, 
                15, 
                6,
                help="Deeper trees can model more complex patterns",
                key="xgb_max_depth"
            )
            
            config['xgboost']['learning_rate'] = st.select_slider(
                "Learning Rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
                value=0.1,
                help="Lower values need more boosting rounds",
                key="xgb_learning_rate"
            )
            
            config['xgboost']['n_estimators'] = st.slider(
                "Number of Trees", 
                50, 
                500, 
                100,
                step=10,
                help="More trees = better performance but slower",
                key="xgb_n_estimators"
            )
            
        with col2:
            config['xgboost']['subsample'] = st.slider(
                "Subsample Ratio", 
                0.5, 
                1.0, 
                0.8,
                step=0.05,
                help="Fraction of samples used for training trees",
                key="xgb_subsample"
            )
            
            config['xgboost']['colsample_bytree'] = st.slider(
                "Column Subsample Ratio", 
                0.5, 
                1.0, 
                0.8,
                step=0.05,
                help="Fraction of features used for training trees",
                key="xgb_colsample"
            )
            
            config['xgboost']['lag_features'] = st.slider(
                "Number of Lag Features", 
                1, 
                12, 
                3,
                help="Previous time steps to use as features",
                key="xgb_lag_features"
            )
    
    # LSTM Configuration Tab
    with config_tabs[3]:
        st.markdown("### LSTM Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            config['lstm']['sequence_length'] = st.slider(
                "Sequence Length", 
                3, 
                24, 
                12,
                help="Number of time steps to look back",
                key="lstm_seq_length"
            )
            
            config['lstm']['epochs'] = st.slider(
                "Training Epochs", 
                20, 
                200, 
                50,
                step=5,
                help="More epochs = better fit but risk of overfitting",
                key="lstm_epochs"
            )
            
        with col2:
            # Use multiselect to get array of units
            lstm_units = st.multiselect(
                "LSTM Layer Units",
                options=[16, 32, 64, 128, 256],
                default=[64, 32],
                help="Units in each LSTM layer (first to last)",
                key="lstm_units"
            )
            config['lstm']['units'] = lstm_units if lstm_units else [64, 32]
            
            config['lstm']['dropout'] = st.slider(
                "Dropout Rate", 
                0.0, 
                0.5, 
                0.2,
                step=0.05,
                help="Higher values = more regularization",
                key="lstm_dropout"
            )
    
    # Ensemble Configuration Tab
    with config_tabs[4]:
        st.markdown("### Ensemble Configuration")
        
        # Which models to include
        available_models = ["Auto ARIMA", "ARIMA", "Exponential Smoothing", "XGBoost", "LSTM", "Prophet"]
        config['ensemble']['models'] = st.multiselect(
            "Models to Include",
            options=available_models,
            default=["Auto ARIMA", "Exponential Smoothing", "XGBoost"],
            help="Select models to include in the ensemble",
            key="ensemble_models"
        )
        
        # Weighting method
        config['ensemble']['method'] = st.radio(
            "Weighting Method",
            options=["Equal Weights", "Weighted by Accuracy", "Optimal Weights"],
            index=0,
            help="How to weight different models in the ensemble",
            key="ensemble_method"
        )
        
        # Training window for accuracy-based weights
        if config['ensemble']['method'] in ["Weighted by Accuracy", "Optimal Weights"]:
            config['ensemble']['validation_window'] = st.slider(
                "Validation Window Size (%)", 
                10, 
                40, 
                20,
                help="Percentage of data to use for validation and weight calculation",
                key="ensemble_validation_window"
            )
    
    # Add a button to apply the advanced configuration
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        apply_config = st.button("Apply Advanced Configuration", type="primary", key="apply_advanced_config")
    
    # Return the config if applied, otherwise None
    if apply_config:
        st.success("✅ Advanced training configuration applied!")
        return config
    else:
        return None

def apply_advanced_training(forecasts, train_data, test_data, target_col, config, forecast_periods, future_index=None):
    """
    Apply advanced training techniques to improve model performance
    
    Args:
        forecasts: Dictionary of existing forecasts
        train_data: Training data DataFrame
        test_data: Test data DataFrame (can be None)
        target_col: Target column name
        config: Advanced configuration dictionary
        forecast_periods: Number of periods to forecast
        future_index: Future index for forecasts
        
    Returns:
        Updated forecasts dictionary with improved models
    """
    # Initialize enhanced forecasts dictionary
    enhanced_forecasts = forecasts.copy()
    
    # Apply advanced Auto ARIMA if available and configured
    if 'Auto ARIMA' in forecasts and 'arima' in config:
        try:
            from modules.demand.auto_arima_mod import generate_auto_arima_forecast
            
            # Log training attempt
            logger.info("Enhancing Auto ARIMA model with advanced training...")
            
            # Get advanced parameters from config
            arima_params = {
                'max_p': config['arima'].get('max_p', 5),
                'max_d': config['arima'].get('max_d', 2),
                'max_q': config['arima'].get('max_q', 5),
                'seasonal': config['arima'].get('seasonal', True),
                'information_criterion': config['arima'].get('information_criterion', 'aic'),
                'stepwise': config['arima'].get('stepwise', True),
                'n_fits': config['arima'].get('n_fits', 50)
            }
            
            # Add seasonal parameters if using seasonality
            if arima_params['seasonal']:
                arima_params.update({
                    'm': config['arima'].get('seasonal_periods', 12),
                    'max_P': config['arima'].get('max_P', 2),
                    'max_D': config['arima'].get('max_D', 1),
                    'max_Q': config['arima'].get('max_Q', 2)
                })
            
            # Generate enhanced forecast
            enhanced_result = generate_auto_arima_forecast(
                train_data=train_data[target_col],
                periods=forecast_periods,
                future_index=future_index,
                **arima_params
            )
            
            # Update the forecast dictionary
            enhanced_forecasts['Auto ARIMA (Enhanced)'] = enhanced_result
            logger.info("Auto ARIMA enhancement completed successfully")
            
        except Exception as e:
            logger.error(f"Error enhancing Auto ARIMA model: {str(e)}")
    
    # Add more model enhancements here (XGBoost, LSTM, etc.)
    
    return enhanced_forecasts
