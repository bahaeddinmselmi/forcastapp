"""
Quick fix script to replace the problematic Auto ARIMA implementation
"""

def fix_auto_arima():
    # Define the corrected implementation
    corrected_auto_arima_impl = """                        # Add new advanced models
                        if 'Auto ARIMA' in models_to_run:
                            try:
                                # Add more robust handling for Auto ARIMA
                                with st.spinner("Running Auto ARIMA model (this may take a moment)..."):
                                    seasonal = config.get('arima', {}).get('seasonal', True) if config.get('advanced_features', False) else True
                                    seasonal_periods = 12  # Default for monthly data
                                    
                                    # Check if we can determine seasonality from the data
                                    if isinstance(train_data.index, pd.DatetimeIndex) and len(train_data) >= 24:
                                        # For longer data, try to infer seasonal periods
                                        freq = pd.infer_freq(train_data.index)
                                        if freq:
                                            if freq.startswith('D') or freq.startswith('B'):  # Daily data
                                                seasonal_periods = 7  # Weekly seasonality
                                            elif freq.startswith('M') or freq.startswith('MS'):  # Monthly data
                                                seasonal_periods = 12  # Yearly seasonality
                                            elif freq.startswith('Q') or freq.startswith('QS'):  # Quarterly data
                                                seasonal_periods = 4  # Yearly seasonality
                                    
                                    st.info(f"Using frequency: MS with seasonal_periods={seasonal_periods} for Auto ARIMA model")
                                    
                                    # Call the advanced implementation with better error handling
                                    try:
                                        # Make the max_seasonal_order include the detected seasonal_periods
                                        if seasonal and seasonal_periods:
                                            # Create max_seasonal_order tuple with the detected periods
                                            max_seasonal_order = (2, 1, 2, seasonal_periods)
                                        else:
                                            max_seasonal_order = (2, 1, 2, 12)  # Default
                                        
                                        auto_arima_result = auto_arima_forecast(
                                            train_data=train_data[target_col],
                                            periods=forecast_periods,
                                            seasonal=seasonal,
                                            max_seasonal_order=max_seasonal_order,
                                            future_index=future_index,
                                            return_conf_int=True
                                        )
                                        
                                        # Make sure the model is properly labeled
                                        if 'model' in auto_arima_result:
                                            auto_arima_result['model'] = 'Auto ARIMA'
                                            
                                        # Verify forecast results are valid
                                        if 'forecast' in auto_arima_result and auto_arima_result['forecast'] is not None:
                                            if len(auto_arima_result['forecast']) > 0:
                                                forecasts['Auto ARIMA'] = auto_arima_result
                                                # Show model information if available
                                                if 'model_order' in auto_arima_result:
                                                    st.success(f"Auto ARIMA selected order: {auto_arima_result['model_order']}")
                                            else:
                                                st.warning("Auto ARIMA generated an empty forecast. Skipping.")
                                        else:
                                            st.warning("Auto ARIMA forecast failed to produce valid results. Skipping.")
                                    except Exception as inner_e:
                                        st.error(f"Error in Auto ARIMA model: {inner_e}")
                                        # Create a basic fallback forecast
                                        try:
                                            fallback_forecast = generate_trend_fallback_forecast(
                                                train_data[target_col], 
                                                forecast_periods, 
                                                future_index
                                            )
                                            forecasts["Auto ARIMA (Fallback)"] = fallback_forecast
                                            st.warning(f"Using fallback trend forecast for Auto ARIMA due to error: {inner_e}")
                                        except Exception as fallback_err:
                                            st.error(f"Even fallback forecast failed: {fallback_err}")
                            except Exception as e:
                                st.error(f"Error in Auto ARIMA forecast: {e}")
"""

    try:
        # Read the file
        with open(r'C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the start and end markers for the Auto ARIMA section
        start_marker = "                        # Add new advanced models\n                        if 'Auto ARIMA' in models_to_run:"
        end_marker = "                                st.error(f\"Error in Auto ARIMA forecast: {e}\")"
        
        # Find the section to replace
        start_pos = content.find(start_marker)
        if start_pos == -1:
            print("Could not find start marker")
            return False
            
        end_pos = content.find(end_marker, start_pos)
        if end_pos == -1:
            print("Could not find end marker")
            return False
        
        # Calculate the end position including the end marker
        end_pos = end_pos + len(end_marker)
        
        # Replace the problematic section
        new_content = content[:start_pos] + corrected_auto_arima_impl + content[end_pos:]
        
        # Write the fixed content back
        with open(r'C:\Users\Public\Downloads\ibp\dd\app\modules\demand\ui.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("Successfully fixed Auto ARIMA implementation")
        return True
    except Exception as e:
        print(f"Error fixing Auto ARIMA implementation: {e}")
        return False

if __name__ == "__main__":
    fix_auto_arima()
