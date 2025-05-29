"""
Fix for Market Intelligence tab to properly detect forecasts.
This script creates a specialized solution for the Market Intelligence feature.
"""
import streamlit as st
import pandas as pd

def debug_session_state_forecasts():
    """Print information about forecasts in session state for debugging"""
    print("\n=== SESSION STATE DEBUG INFO ===")
    
    if 'forecasts' in st.session_state:
        print(f"Forecasts found in session state: {len(st.session_state['forecasts'])} models")
        for model_name, model_data in st.session_state['forecasts'].items():
            print(f"- Model: {model_name}")
            if 'forecast' in model_data:
                print(f"  - Has forecast data: {type(model_data['forecast'])}")
                print(f"  - Length: {len(model_data['forecast']) if hasattr(model_data['forecast'], '__len__') else 'N/A'}")
            else:
                print("  - No forecast data")
    else:
        print("No forecasts in session state")
        
    if 'best_model' in st.session_state:
        print(f"Best model: {st.session_state['best_model']}")
    else:
        print("No best model selected")
    
    print("===========================\n")

def fix_market_intelligence_detection():
    """
    Fix for market intelligence to detect forecasts properly
    Call this function in the app before checking forecasts
    """
    # Check if we have actual forecast models but not in correct format
    if 'forecasts' in st.session_state and st.session_state['forecasts']:
        models_with_forecasts = {}
        
        # Check if we have data in unexpected format
        for model_name, model_data in st.session_state['forecasts'].items():
            if isinstance(model_data, dict) and 'forecast' in model_data and model_data['forecast'] is not None:
                # This is a valid forecast model
                models_with_forecasts[model_name] = model_data
                
        if models_with_forecasts and ('best_model' not in st.session_state):
            # If we have models but no best model, set the first one as best
            st.session_state['best_model'] = next(iter(models_with_forecasts.keys()))
            print(f"Set best model to: {st.session_state['best_model']}")
            
        # Debug info
        debug_session_state_forecasts()
            
    return

# Add a message to guide implementation
print("Add this function call to the Market Intelligence tab section:")
print("fix_market_intelligence_detection()")
print("This will fix forecast detection in the Market Intelligence tab")
