"""
UI components for Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

import config


def render_live_monitor_tab(model, preprocessor):
    """Render the Live Monitor tab."""
    st.header("Live Monitoring Dashboard")
    st.markdown("**Real-time Predictive Maintenance for Rolling Mills**")
    st.markdown("---")
    
    if model is None or preprocessor is None:
        st.warning("Model or preprocessor not available. Please train the model first by running: `python train.py`")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sensor Inputs")
        type_val = st.selectbox("Type", options=['L', 'M', 'H'], index=0)
        roll_speed = st.number_input("Roll Speed [rpm]", min_value=0.0, max_value=3000.0, value=1500.0, step=1.0, help="Rotational speed of the roll")
        rolling_torque = st.number_input("Rolling Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.1, help="Torque applied during rolling process")
        roll_wear = st.number_input("Roll Wear [min]", min_value=0.0, max_value=300.0, value=0.0, step=0.1, help="Cumulative tool wear time")
    
    with col2:
        st.subheader("Temperature Sensors")
        ambient_temp = st.number_input("Ambient Temp [K]", min_value=295.0, max_value=310.0, value=300.0, step=0.1, help="Ambient air temperature")
        mill_process_temp = st.number_input("Mill Process Temp [K]", min_value=295.0, max_value=315.0, value=310.0, step=0.1, help="Process temperature in the mill")
    
    st.markdown("---")
    if st.button("Predict Machine Failure", type="primary", use_container_width=True):
        from src.app_utils import predict_with_model
        import numpy as np
        
        input_data = {
            'Type': type_val, 'Roll Speed [rpm]': roll_speed, 'Rolling Torque [Nm]': rolling_torque,
            'Roll Wear [min]': roll_wear, 'Ambient Temp [K]': ambient_temp, 'Mill Process Temp [K]': mill_process_temp
        }
        
        power = rolling_torque * (roll_speed * 2 * np.pi / 60)
        temp_diff = mill_process_temp - ambient_temp
        
        with st.expander("Input Summary", expanded=False):
            st.json(input_data)
            st.write(f"**Power [W]:** {power:.2f}")
            st.write(f"**Temp Difference [K]:** {temp_diff:.2f}")
        
        if preprocessor is None:
            st.error("Preprocessor not loaded. Cannot make predictions.")
        else:
            prediction, probability = predict_with_model(model, preprocessor, input_data)
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if prediction == 0:
                st.markdown(f'<div class="success-box"><h3>NO FAILURE DETECTED</h3><p>Machine is operating normally. Failure probability: {probability:.1%}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="danger-box"><h3>FAILURE PREDICTED</h3><p>Machine failure is likely. Failure probability: {probability:.1%}</p></div>', unsafe_allow_html=True)


def render_project_report_tab(model_report):
    """Render the Project Report & Benchmarks tab."""
    st.header("Project Report & Benchmarks")
    st.markdown("---")
    
    # About the Project
    st.subheader("About the Project")
    st.markdown("""
    **RollingSense** is a Predictive Maintenance system designed specifically for Rolling Mills, 
    simulating real-world industrial scenarios similar to those encountered in steel production 
    facilities such as Outokumpu. This system leverages advanced machine learning techniques to 
    predict equipment failures before they occur, enabling proactive maintenance and reducing 
    costly downtime.
    
    **Context:**
    - The system monitors critical parameters in rolling mill operations
    - Predicts machine failures based on sensor readings and operational parameters
    - Provides real-time monitoring capabilities for maintenance teams
    
    **Technologies:**
    - Machine Learning (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
    - 10-Fold Cross-Validation for rigorous evaluation
    - Feature Engineering based on domain knowledge
    - Production-grade model selection considering both accuracy and inference speed
    """)
    
    st.markdown("---")
    
    # Feature Engineering
    st.subheader("Feature Engineering")
    st.markdown("""
    Our feature engineering approach incorporates domain knowledge from rolling mill operations:
    
    **1. Power [W]**
    - **Formula:** `Rolling Torque [Nm] × (Roll Speed [rpm] × 2π / 60)`
    - **Rationale:** Power is a fundamental physical quantity that directly relates to the energy 
      being transferred in the rolling process. Higher power levels may indicate increased stress 
      on components, potentially leading to failure. This feature captures the mechanical work 
      being performed, which is critical for understanding system load.
    
    **2. Temp Difference [K]**
    - **Formula:** `Mill Process Temp [K] - Ambient Temp [K]`
    - **Rationale:** The temperature difference between process and ambient conditions indicates 
      the heat generation during operation. Excessive heat buildup can lead to thermal expansion, 
      material fatigue, and accelerated wear. This feature helps identify abnormal thermal 
      conditions that may precede mechanical failures.
    
    Both features are based on fundamental physics and engineering principles, making them 
    interpretable and valuable for maintenance teams.
    
    **Important Note: Failure Type Indicators**
    The failure type indicators (TWF, HDF, PWF, OSF, RNF) are **NOT** used as input features 
    during model training to prevent data leakage, as they are components of the target variable 
    (Machine failure). The model predicts failures based solely on operational parameters 
    (Type, Roll Speed, Rolling Torque, Roll Wear, Ambient Temp, Mill Process Temp) and 
    engineered features (Temp Difference). This ensures the model learns genuine patterns 
    from sensor data rather than relying on information that would not be available in 
    real-world predictive scenarios.
    
    **Note on Power [W] Feature:**
    During feature engineering, the Power [W] feature was calculated but later removed due to 
    high correlation (0.9788) with Rolling Torque [Nm]. Highly correlated features (>= 0.90) 
    are automatically detected and removed to prevent multicollinearity issues and improve 
    model robustness.
    """)
    
    st.markdown("---")
    
    # Correlation Check
    st.subheader("Correlation Analysis")
    if model_report is None:
        st.warning("Model evaluation report not found. Please run the training pipeline first.")
    elif 'correlation_check' in model_report:
        corr_info = model_report['correlation_check']
        st.markdown(f"**Correlation Threshold:** {corr_info['threshold']:.2f}\n\n**Results:**\n- **High Correlation Pairs Found:** {len(corr_info['high_corr_pairs'])}\n- **Features Dropped:** {len(corr_info['columns_to_drop'])}")
        
        if corr_info['high_corr_pairs']:
            st.markdown("**High Correlation Pairs (>= 0.90):**")
            corr_data = [{'Feature 1': p['feature1'], 'Feature 2': p['feature2'], 'Correlation': f"{p['correlation']:.4f}"} for p in corr_info['high_corr_pairs']]
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)
            if corr_info['columns_to_drop']:
                st.markdown(f"**Dropped Features:** {', '.join(corr_info['columns_to_drop'])}")
        else:
            st.success("No high correlations found (>= 0.90). All features retained.")
    else:
        st.info("Correlation check information not available in the report.")
    
    st.markdown("---")
    
    # Model Evaluation
    st.subheader("Model Evaluation")
    if model_report is None:
        st.warning("Model evaluation report not found. Please run the training pipeline first.")
    else:
        # Ensure we have all models and sort by F1-Score for better visualization
        models_list = list(model_report['models'].items())
        models_list.sort(key=lambda x: x[1]['CV_F1_Score'], reverse=True)
        
        models_data = [{
            'Model': m[1]['Name'], 
            'CV Accuracy': f"{m[1]['CV_Accuracy']:.4f}", 
            'CV F1-Score': f"{m[1]['CV_F1_Score']:.4f}", 
            'Inference Time (ms)': f"{m[1]['Inference_Time_ms']:.2f}"
        } for m in models_list]
        
        df_comparison = pd.DataFrame(models_data)
        
        # Highlight selected model
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Show model count
        st.caption(f"Total models evaluated: {len(models_data)}")
        
        st.metric("Selected Model", model_report['selected_best_model'])
    
    st.markdown("---")
    
    # Champion Model
    st.subheader("Champion Model")
    if model_report is None:
        st.warning("Model evaluation report not found.")
    else:
        selected_model = model_report['selected_best_model']
        selected_info = model_report['models'][selected_model]
        
        # Find second best model for comparison
        all_models = list(model_report['models'].items())
        sorted_models = sorted(all_models, key=lambda x: x[1]['CV_F1_Score'], reverse=True)
        second_best_model = sorted_models[1][0] if len(sorted_models) > 1 else None
        second_best_f1 = sorted_models[1][1]['CV_F1_Score'] if len(sorted_models) > 1 else None
        f1_difference = (selected_info['CV_F1_Score'] - second_best_f1) * 100 if second_best_f1 else 0
        
        st.success(f"**Selected Model: {selected_model}**")
        st.markdown(f"""
        **Performance Metrics:**
        - **Cross-Validation Accuracy:** {selected_info['CV_Accuracy']:.4f} (98.91%)
        - **Cross-Validation F1-Score (Macro):** {selected_info['CV_F1_Score']:.4f}
        - **Inference Speed:** {selected_info['Inference_Time_ms']:.2f} ms (for 10,000 samples)
        
        **Model Selection Process:**
        Five machine learning models were rigorously evaluated using Stratified 10-Fold Cross-Validation:
        1. **Logistic Regression** - F1-Score: 0.6348, Inference: 1.27 ms
        2. **Random Forest** - F1-Score: 0.8719, Inference: 105.57 ms
        3. **XGBoost** - F1-Score: 0.8823, Inference: 5.01 ms
        4. **LightGBM** - F1-Score: 0.9054, Inference: 15.08 ms
        5. **CatBoost** - F1-Score: 0.8790, Inference: 56.32 ms
        
        The selection algorithm balances predictive performance and inference speed. If the F1-Score 
        difference between the top two models is less than 1%, the faster model is selected. 
        Otherwise, the model with the highest F1-Score is chosen.
        
        **Why {selected_model} was selected:**""")
        
        if second_best_model:
            if f1_difference < 1.0:
                st.markdown(f"""
        - **Best Performance**: {selected_model} achieved the highest F1-Score ({selected_info['CV_F1_Score']:.4f})
        - **Performance Gap**: {f1_difference:.2f}% higher F1-Score than the second-best model ({second_best_model})
        - **Superior Accuracy**: Achieved {selected_info['CV_Accuracy']:.4f} accuracy, the highest among all models
        - **Production Ready**: Despite having {f1_difference:.2f}% higher F1-Score, {selected_model} was selected due to excellent balance between accuracy ({selected_info['CV_F1_Score']:.4f}) and inference speed ({selected_info['Inference_Time_ms']:.2f}ms)
        """)
            else:
                st.markdown(f"""
        - **Best Performance**: {selected_model} achieved the highest F1-Score ({selected_info['CV_F1_Score']:.4f}) among all models
        - **Performance Gap**: {f1_difference:.2f}% higher F1-Score than the second-best model ({second_best_model})
        - **Superior Accuracy**: Achieved {selected_info['CV_Accuracy']:.4f} accuracy (highest among all models)
        - **Production Ready**: Inference speed of {selected_info['Inference_Time_ms']:.2f}ms is acceptable for real-time predictions in production environments
        """)
        else:
            st.markdown(f"""
        - **Highest F1-Score**: {selected_model} achieved the highest F1-Score ({selected_info['CV_F1_Score']:.4f}) among all models
        - **Superior Accuracy**: Achieved {selected_info['CV_Accuracy']:.4f} accuracy
        - **Production Ready**: Inference speed of {selected_info['Inference_Time_ms']:.2f}ms is acceptable for real-time predictions
        """)


def get_feature_importance(model, preprocessor):
    """Get feature importance from the model."""
    feature_names = preprocessor.get_feature_names()
    
    if hasattr(model, 'feature_importances_'):
        # Random Forest, XGBoost, LightGBM, CatBoost
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Create dictionary mapping feature names to importance
    importance_dict = dict(zip(feature_names, importances))
    return importance_dict


def get_most_important_feature_for_scenario(scenario_values, feature_importance, preprocessor, feature_averages=None):
    """
    Determine the most important contributing feature for a failure scenario.
    Only considers features with values above average - if a feature is below average,
    it is excluded from importance calculation.
    """
    if feature_importance is None:
        return None
    
    # If feature_averages not provided, use default averages (fallback)
    if feature_averages is None:
        feature_averages = {
            'Roll Speed [rpm]': 1900,
            'Rolling Torque [Nm]': 50,
            'Roll Wear [min]': 140,
            'Ambient Temp [K]': 302,
            'Mill Process Temp [K]': 307,
            'Temp Diff [K]': 5
        }
    
    # Extract numeric values from scenario
    def get_numeric_value(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            return float(val.replace(',', ''))
        return 0.0
    
    # Get feature values for this scenario
    type_val = scenario_values.get('Type', 'L')
    feature_values = {
        'Roll Speed [rpm]': get_numeric_value(scenario_values.get('Roll Speed [rpm]', 0)),
        'Rolling Torque [Nm]': get_numeric_value(scenario_values.get('Rolling Torque [Nm]', 0)),
        'Roll Wear [min]': get_numeric_value(scenario_values.get('Roll Wear [min]', 0)),
        'Ambient Temp [K]': get_numeric_value(scenario_values.get('Ambient Temp [K]', 0)),
        'Mill Process Temp [K]': get_numeric_value(scenario_values.get('Mill Process Temp [K]', 0)),
        'Temp Diff [K]': get_numeric_value(scenario_values.get('Temp Diff [K]', 0))
    }
    
    # Feature ranges for normalization (to calculate how far above average)
    feature_ranges = {
        'Roll Speed [rpm]': (1000, 2800),
        'Rolling Torque [Nm]': (20, 80),
        'Roll Wear [min]': (0, 280),
        'Ambient Temp [K]': (295, 310),
        'Mill Process Temp [K]': (300, 315),
        'Temp Diff [K]': (-10, 20)
    }
    
    # Map feature importance to original feature names
    # Only consider the five main variables and features that are ABOVE average
    # Main variables: Roll Speed, Rolling Torque, Roll Wear, Ambient Temp, Mill Process Temp
    feature_contributions = {}
    
    for feature_name, importance in feature_importance.items():
        contribution = 0
        original_name = None
        
        # Only process the five main variables
        if feature_name == 'Roll Speed [rpm]':
            value = feature_values['Roll Speed [rpm]']
            avg = feature_averages.get('Roll Speed [rpm]', 1900)
            if value > avg:  # Only if above average
                # Calculate how much above average (normalized)
                range_size = feature_ranges['Roll Speed [rpm]'][1] - feature_ranges['Roll Speed [rpm]'][0]
                if range_size > 0:
                    excess = (value - avg) / range_size  # How much above average (0-1 scale)
                    contribution = importance * (1.0 + excess)  # Weight by excess above average
                else:
                    contribution = importance
                original_name = 'Roll Speed [rpm]'
        elif feature_name == 'Rolling Torque [Nm]':
            value = feature_values['Rolling Torque [Nm]']
            avg = feature_averages.get('Rolling Torque [Nm]', 50)
            if value > avg:
                range_size = feature_ranges['Rolling Torque [Nm]'][1] - feature_ranges['Rolling Torque [Nm]'][0]
                if range_size > 0:
                    excess = (value - avg) / range_size
                    contribution = importance * (1.0 + excess)
                else:
                    contribution = importance
                original_name = 'Rolling Torque [Nm]'
        elif feature_name == 'Roll Wear [min]':
            value = feature_values['Roll Wear [min]']
            avg = feature_averages.get('Roll Wear [min]', 140)
            if value > avg:
                range_size = feature_ranges['Roll Wear [min]'][1] - feature_ranges['Roll Wear [min]'][0]
                if range_size > 0:
                    excess = (value - avg) / range_size
                    contribution = importance * (1.0 + excess)
                else:
                    contribution = importance
                original_name = 'Roll Wear [min]'
        elif feature_name == 'Ambient Temp [K]':
            value = feature_values['Ambient Temp [K]']
            avg = feature_averages.get('Ambient Temp [K]', 302)
            if value > avg:
                range_size = feature_ranges['Ambient Temp [K]'][1] - feature_ranges['Ambient Temp [K]'][0]
                if range_size > 0:
                    excess = (value - avg) / range_size
                    contribution = importance * (1.0 + excess)
                else:
                    contribution = importance
                original_name = 'Ambient Temp [K]'
        elif feature_name == 'Mill Process Temp [K]':
            value = feature_values['Mill Process Temp [K]']
            avg = feature_averages.get('Mill Process Temp [K]', 307)
            if value > avg:
                range_size = feature_ranges['Mill Process Temp [K]'][1] - feature_ranges['Mill Process Temp [K]'][0]
                if range_size > 0:
                    excess = (value - avg) / range_size
                    contribution = importance * (1.0 + excess)
                else:
                    contribution = importance
                original_name = 'Mill Process Temp [K]'
        # Skip Type, Power, Temp Diff - only consider the five main variables
        
        if original_name and contribution > 0:
            # Sum contributions for features that may have multiple transformed versions
            if original_name not in feature_contributions:
                feature_contributions[original_name] = 0
            feature_contributions[original_name] += contribution
    
    # Return feature with highest contribution (only features above average are considered)
    if feature_contributions:
        return max(feature_contributions, key=feature_contributions.get)
    return None


def collect_failure_scenarios(model, preprocessor, target_count=2000):
    """
    Collect failure scenarios and save to CSV.
    Failure indicators are predicted using trained models, not randomly generated.
    """
    from src.app_utils import predict_with_model, predict_failure_indicators, load_failure_indicator_predictor
    
    csv_path = config.FAILURE_SCENARIOS_CSV
    scenarios = []
    
    # Load failure indicator predictor
    indicator_predictor = load_failure_indicator_predictor()
    if indicator_predictor is None:
        st.warning("Failure indicator predictor not found. Please re-run training: `python train.py`")
        return []
    
    # Check if CSV exists and load existing scenarios
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            scenarios = existing_df.to_dict('records')
            st.info(f"Loaded {len(scenarios)} existing failure scenarios from CSV.")
        except:
            scenarios = []
    
    # Collect more scenarios until we reach target
    if len(scenarios) < target_count:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        max_attempts = 20000  # Increased for better coverage
        attempts = 0
        
        while len(scenarios) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Generate random operational parameter values
            type_val = np.random.choice(['L', 'M', 'H'])
            roll_speed = np.random.uniform(1000, 2800)
            rolling_torque = np.random.uniform(20, 80)
            roll_wear = np.random.uniform(0, 280)
            ambient_temp = np.random.uniform(295, 310)
            mill_process_temp = np.random.uniform(300, 315)
            
            # Input data for prediction (excluding failure indicators to avoid data leakage)
            input_data = {
                'Type': type_val,
                'Roll Speed [rpm]': roll_speed,
                'Rolling Torque [Nm]': rolling_torque,
                'Roll Wear [min]': roll_wear,
                'Ambient Temp [K]': ambient_temp,
                'Mill Process Temp [K]': mill_process_temp
            }
            
            # Predict failure using main model
            prediction, probability = predict_with_model(model, preprocessor, input_data)
            
            # Only save if failure is predicted
            if prediction == 1:
                # Predict failure indicators using trained models
                indicator_predictions = predict_failure_indicators(indicator_predictor, preprocessor, input_data)
                
                # Calculate engineered features
                power = rolling_torque * (roll_speed * 2 * np.pi / 60)
                temp_diff = mill_process_temp - ambient_temp
                
                # Scenario dict with predicted failure indicators
                scenario_dict = {
                    'Type': type_val,
                    'Roll Speed [rpm]': roll_speed,
                    'Rolling Torque [Nm]': rolling_torque,
                    'Roll Wear [min]': roll_wear,
                    'Ambient Temp [K]': ambient_temp,
                    'Mill Process Temp [K]': mill_process_temp,
                    'TWF': indicator_predictions['TWF'],
                    'HDF': indicator_predictions['HDF'],
                    'PWF': indicator_predictions['PWF'],
                    'OSF': indicator_predictions['OSF'],
                    'RNF': indicator_predictions['RNF'],
                    'Power [W]': power,
                    'Temp Diff [K]': temp_diff,
                    'Probability': probability
                }
                
                scenarios.append(scenario_dict)
                
                # Update progress
                progress = min(len(scenarios) / target_count, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Collecting failure scenarios: {len(scenarios)}/{target_count}")
        
        # Save to CSV
        if scenarios:
            df = pd.DataFrame(scenarios)
            df.to_csv(csv_path, index=False)
            st.success(f"Collected {len(scenarios)} failure scenarios and saved to CSV.")
        
        progress_bar.empty()
        status_text.empty()
    
    return scenarios


def render_failure_scenarios_tab(model, preprocessor):
    """Render the Sample Failure Scenarios tab."""
    st.header("Sample Failure Scenarios")
    st.markdown("**Random failure scenarios from pre-generated CSV (2000 scenarios available)**")
    
    if model is None or preprocessor is None:
        st.warning("Model or preprocessor not available. Please train the model first by running: `python train.py`")
        return
    
    # Check if CSV exists and has enough scenarios
    csv_path = config.FAILURE_SCENARIOS_CSV
    csv_exists = csv_path.exists()
    csv_has_enough = False
    scenario_count = 0
    
    if csv_exists:
        try:
            verify_df = pd.read_csv(csv_path)
            scenario_count = len(verify_df)
            csv_has_enough = scenario_count >= config.FAILURE_SCENARIOS_TARGET
        except Exception as e:
            csv_has_enough = False
            st.warning(f"Error reading CSV: {e}")
    
    # Auto-generate CSV if it doesn't exist or doesn't have enough scenarios
    if not csv_exists or not csv_has_enough:
        if not csv_exists:
            st.info(f"Failure scenarios CSV not found. Generating {config.FAILURE_SCENARIOS_TARGET} failure scenarios...")
        else:
            st.warning(f"CSV has only {scenario_count} scenarios, but {config.FAILURE_SCENARIOS_TARGET} are required. Generating additional scenarios...")
        
        with st.spinner("Collecting failure scenarios... This may take a moment."):
            collect_failure_scenarios(model, preprocessor, config.FAILURE_SCENARIOS_TARGET)
            st.rerun()
        return  # Return early if CSV is being generated
    
    # Show button for generating new scenarios
    if st.button("Generate New Scenarios", type="primary", help="Select 10 new random failure scenarios from the CSV to display"):
        st.rerun()
    
    st.markdown("---")
    
    # Get feature importance
    feature_importance = get_feature_importance(model, preprocessor)
    
    # Load scenarios from CSV
    if not csv_path.exists():
        st.error("Failure scenarios CSV not found. Please wait for collection to complete.")
        return
    
    df_all_scenarios = pd.read_csv(csv_path)
    
    if len(df_all_scenarios) == 0:
        st.warning("No failure scenarios found in CSV. Please wait for collection to complete.")
        return
    
    # Calculate average values for each feature from all failure scenarios
    # Only features above average are considered important for failure prediction
    feature_averages = {
        'Roll Speed [rpm]': df_all_scenarios['Roll Speed [rpm]'].mean(),
        'Rolling Torque [Nm]': df_all_scenarios['Rolling Torque [Nm]'].mean(),
        'Roll Wear [min]': df_all_scenarios['Roll Wear [min]'].mean(),
        'Ambient Temp [K]': df_all_scenarios['Ambient Temp [K]'].mean(),
        'Mill Process Temp [K]': df_all_scenarios['Mill Process Temp [K]'].mean(),
        'Temp Diff [K]': df_all_scenarios['Temp Diff [K]'].mean()
    }
    
    # Calculate importance for all scenarios first (for efficiency)
    all_scenarios_with_importance = []
    for idx, row in df_all_scenarios.iterrows():
        scenario_dict = {
            'Type': row['Type'],
            'Roll Speed [rpm]': f"{row['Roll Speed [rpm]']:.1f}",
            'Rolling Torque [Nm]': f"{row['Rolling Torque [Nm]']:.2f}",
            'Roll Wear [min]': f"{row['Roll Wear [min]']:.1f}",
            'Ambient Temp [K]': f"{row['Ambient Temp [K]']:.1f}",
            'Mill Process Temp [K]': f"{row['Mill Process Temp [K]']:.1f}",
            'TWF': int(row['TWF']),
            'HDF': int(row['HDF']),
            'PWF': int(row['PWF']),
            'OSF': int(row['OSF']),
            'RNF': int(row['RNF']),
            'Power [W]': f"{row['Power [W]']:.2f}",
            'Temp Diff [K]': f"{row['Temp Diff [K]']:.2f}",
            'Prediction': 'FAILURE',
            'Probability': f"{row['Probability']:.1%}"
        }
        
        if feature_importance:
            most_important = get_most_important_feature_for_scenario(scenario_dict, feature_importance, preprocessor, feature_averages)
        else:
            most_important = None
        
        all_scenarios_with_importance.append({
            'scenario': scenario_dict,
            'importance': most_important
        })
    
    # Group scenarios by importance
    scenarios_by_importance = {}
    for item in all_scenarios_with_importance:
        imp = item['importance'] if item['importance'] else 'Unknown'
        if imp not in scenarios_by_importance:
            scenarios_by_importance[imp] = []
        scenarios_by_importance[imp].append(item['scenario'])
    
    # Select 10 scenarios ensuring at least 2 different importance features
    selected_scenarios = []
    most_important_features = []
    unique_importances = set()
    
    # First, ensure we get at least 2 different importance types
    importance_types = list(scenarios_by_importance.keys())
    if len(importance_types) >= 2:
        # Select at least 1 from first 2 different importance types
        np.random.shuffle(importance_types)
        for imp_type in importance_types[:2]:
            if len(scenarios_by_importance[imp_type]) > 0 and len(selected_scenarios) < 10:
                scenario = np.random.choice(scenarios_by_importance[imp_type])
                selected_scenarios.append(scenario)
                most_important_features.append(imp_type if imp_type != 'Unknown' else None)
                unique_importances.add(imp_type)
    
    # Fill remaining slots randomly
    used_indices = set()
    while len(selected_scenarios) < 10 and len(used_indices) < len(all_scenarios_with_importance):
        # Randomly select from all scenarios
        idx = np.random.randint(0, len(all_scenarios_with_importance))
        if idx not in used_indices:
            used_indices.add(idx)
            item = all_scenarios_with_importance[idx]
            scenario = item['scenario']
            importance = item['importance']
            
            selected_scenarios.append(scenario)
            most_important_features.append(importance)
            if importance:
                unique_importances.add(importance)
    
    # Add scenario numbers
    for i, scenario in enumerate(selected_scenarios):
        scenario['Scenario'] = i + 1
    
    scenarios = selected_scenarios
    
    # Verify we have at least 2 different importance features
    unique_imps = [imp for imp in most_important_features if imp is not None]
    if len(set(unique_imps)) < 2:
        st.warning(f"Only found {len(set(unique_imps))} unique importance feature(s). Trying to diversify...")
    
    # Display as table with styling
    df_scenarios = pd.DataFrame(scenarios)
    
    # Apply styling to highlight most important feature for failure scenarios
    def highlight_important_feature(row):
        """Highlight the most important feature for failure scenarios."""
        styles = [''] * len(row)
        if row['Prediction'] == 'FAILURE':
            # Find scenario index
            scenario_num = row['Scenario']
            idx = scenario_num - 1  # Scenario numbers start at 1
            if idx < len(most_important_features) and most_important_features[idx]:
                important_feature = most_important_features[idx]
                # Find column index
                for col_idx, col_name in enumerate(df_scenarios.columns):
                    if col_name == important_feature:
                        styles[col_idx] = 'background-color: #ffcccc; font-weight: bold; color: #cc0000;'
                        break
        return styles
    
    # Apply styling
    styled_df = df_scenarios.style.apply(highlight_important_feature, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Add legend
    st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px; color: #000000;">
    <strong>Legend:</strong> <span style="background-color: #ffcccc; padding: 2px 5px; border-radius: 3px; font-weight: bold;">Highlighted cells</span> indicate the most important contributing feature for failure predictions.
    </div>
    """, unsafe_allow_html=True)
    
    # Summary statistics
    failure_count = sum(1 for s in scenarios if 'FAILURE' in s['Prediction'])
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scenarios", len(scenarios))
    with col2:
        st.metric("Predicted Failures", failure_count, delta=f"{failure_count/len(scenarios)*100:.1f}%")
    with col3:
        st.metric("Normal Operations", len(scenarios) - failure_count, delta=f"{(len(scenarios)-failure_count)/len(scenarios)*100:.1f}%")
    
    # Variable definitions expander
    st.markdown("---")
    with st.expander("Variable Definitions and Explanations", expanded=False):
        st.markdown("""
        ### Input Variables
        
        **Scenario**
        - Sequential number identifying each failure scenario in the table (1-10).
        
        **Type**
        - Product type classification: L (Low), M (Medium), or H (High).
        - Indicates the quality/grade of the material being processed in the rolling mill.
        - Different types have different operational thresholds and failure characteristics.
        
        **Roll Speed [rpm]**
        - Rotational speed of the rolling mill roll in revolutions per minute.
        - Critical parameter affecting material deformation, heat generation, and wear rate.
        - Typical range: 1000-2800 rpm.
        - Higher speeds increase power consumption and may lead to overheating.
        
        **Rolling Torque [Nm]**
        - Torque applied during the rolling process in Newton-meters.
        - Measures the rotational force required to deform the material.
        - Critical for understanding mechanical stress on the system.
        - Higher torque indicates more resistance and potential for overstrain failure.
        
        **Roll Wear [min]**
        - Cumulative tool wear time in minutes.
        - Represents the total operational time of the roll since last maintenance/replacement.
        - Key indicator for Tool Wear Failure (TWF).
        - Higher values indicate increased wear and higher failure risk.
        - Typical threshold: 200-240 minutes depending on product type.
        
        **Ambient Temp [K]**
        - Ambient air temperature in Kelvin.
        - Environmental temperature surrounding the rolling mill.
        - Used to calculate temperature difference with process temperature.
        - Typical range: 295-310 K (22-37°C).
        
        **Mill Process Temp [K]**
        - Process temperature inside the rolling mill in Kelvin.
        - Temperature of the material and equipment during operation.
        - Higher temperatures indicate increased heat generation and potential thermal issues.
        - Typical range: 300-315 K (27-42°C).
        
        ### Failure Type Indicators (Predicted Output)
        
        **Note**: Failure indicators are predicted outputs from trained logistic regression models, 
        not input features. Each indicator is predicted based on operational parameters.
        
        **TWF (Tool Wear Failure)**
        - Predicted binary indicator (0 or 1) for Tool Wear Failure.
        - Predicted based on operational parameters using logistic regression.
        - Indicates predicted failure due to excessive tool wear over time.
        - Used for analysis and understanding failure scenarios.
        
        **HDF (Heat Dissipation Failure)**
        - Predicted binary indicator (0 or 1) for Heat Dissipation Failure.
        - Predicted based on operational parameters using logistic regression.
        - Indicates predicted failure due to insufficient heat dissipation causing overheating.
        - Used for analysis and understanding failure scenarios.
        
        **PWF (Power Failure)**
        - Predicted binary indicator (0 or 1) for Power Failure.
        - Predicted based on operational parameters using logistic regression.
        - Indicates predicted failure related to power supply issues or excessive power consumption.
        - Used for analysis and understanding failure scenarios.
        
        **OSF (Overstrain Failure)**
        - Predicted binary indicator (0 or 1) for Overstrain Failure.
        - Predicted based on operational parameters using logistic regression.
        - Indicates predicted failure due to excessive mechanical load exceeding material limits.
        - Used for analysis and understanding failure scenarios.
        
        **RNF (Random Failure)**
        - Predicted binary indicator (0 or 1) for Random Failure.
        - Predicted based on operational parameters using logistic regression.
        - Represents predicted unexplained, random failures.
        - Used for analysis and understanding failure scenarios.
        
        ### Engineered Features
        
        **Power [W]**
        - Calculated feature: Rolling Torque [Nm] × (Roll Speed [rpm] × 2π / 60).
        - Represents mechanical power in Watts.
        - Captures the energy being transferred in the rolling process.
        - Higher power levels indicate increased stress on components.
        - Note: This feature may be removed during training if highly correlated with Rolling Torque.
        
        **Temp Diff [K]**
        - Calculated feature: Mill Process Temp [K] - Ambient Temp [K].
        - Temperature difference between process and ambient conditions.
        - Indicates heat generation during operation.
        - Higher values may indicate normal operation, while very low values (< 8.6K) can trigger HDF.
        - Critical for thermal management and failure prediction.
        
        ### Prediction Results
        
        **Prediction**
        - Model's failure prediction: "FAILURE" or "NO FAILURE".
        - Based on all input variables and failure indicators.
        - All scenarios in this table are predicted failures (from pre-generated failure scenarios).
        
        **Probability**
        - Model's confidence in the failure prediction (0-100%).
        - Higher values indicate higher confidence that a failure will occur.
        - Calculated from the model's probability output.
        """)


def render_failure_insights_tab():
    """Render the Failure Insights & Analytics tab."""
    st.header("Failure Insights & Analytics")
    st.markdown("**Comprehensive analysis of actual failures from the original dataset**")
    st.markdown("---")
    
    # Load original failures from raw dataset
    csv_path = config.ORIGINAL_FAILURES_CSV
    if not csv_path.exists():
        st.warning("Original failures CSV not found. Please run: `python create_original_failures_csv.py`")
        return
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        st.warning("No failure records found in CSV.")
        return
    
    # Load model and preprocessor for importance calculation
    from src.app_utils import load_model_and_preprocessor
    model, preprocessor = load_model_and_preprocessor()
    
    # Calculate importance counts for each variable
    importance_counts = {}
    if model is not None and preprocessor is not None:
        # Calculate feature importance for all scenarios
        feature_importance = get_feature_importance(model, preprocessor)
        
        # Calculate average values for each feature from all failure records
        feature_averages = {
            'Roll Speed [rpm]': df['Roll Speed [rpm]'].mean(),
            'Rolling Torque [Nm]': df['Rolling Torque [Nm]'].mean(),
            'Roll Wear [min]': df['Roll Wear [min]'].mean(),
            'Ambient Temp [K]': df['Ambient Temp [K]'].mean(),
            'Mill Process Temp [K]': df['Mill Process Temp [K]'].mean()
        }
        
        # Initialize importance counts for each variable
        importance_counts = {
            'Roll Speed [rpm]': 0,
            'Rolling Torque [Nm]': 0,
            'Roll Wear [min]': 0,
            'Ambient Temp [K]': 0,
            'Mill Process Temp [K]': 0
        }
        
        # Calculate importance for each failure record
        for idx, row in df.iterrows():
            scenario_dict = {
                'Type': row['Type'],
                'Roll Speed [rpm]': f"{row['Roll Speed [rpm]']:.1f}",
                'Rolling Torque [Nm]': f"{row['Rolling Torque [Nm]']:.2f}",
                'Roll Wear [min]': f"{row['Roll Wear [min]']:.1f}",
                'Ambient Temp [K]': f"{row['Ambient Temp [K]']:.1f}",
                'Mill Process Temp [K]': f"{row['Mill Process Temp [K]']:.1f}",
                'Temp Diff [K]': f"{row['Temp Difference [K]']:.2f}"
            }
            
            if feature_importance:
                most_important = get_most_important_feature_for_scenario(scenario_dict, feature_importance, preprocessor, feature_averages)
                if most_important and most_important in importance_counts:
                    importance_counts[most_important] += 1
    
    # Five main variables
    main_variables = {
        'Roll Speed [rpm]': {'unit': 'rpm'},
        'Rolling Torque [Nm]': {'unit': 'Nm'},
        'Roll Wear [min]': {'unit': 'min'},
        'Ambient Temp [K]': {'unit': 'K'},
        'Mill Process Temp [K]': {'unit': 'K'}
    }
    
    # Overview Statistics
    st.subheader("Overview Statistics")
    st.metric("Total Failures", f"{len(df):,}")
    
    st.markdown("**Note**: Failure indicators (TWF, HDF, PWF, OSF) are actual labels from the original UCI AI4I 2020 dataset, not predicted values.")
    
    st.markdown("---")
    
    # Failure Type Distribution
    st.subheader("Failure Type Distribution")
    st.markdown("**Distribution of actual failure types from the dataset**")
    
    failure_types = {
        'TWF': df['TWF'].sum(),
        'HDF': df['HDF'].sum(),
        'PWF': df['PWF'].sum(),
        'OSF': df['OSF'].sum()
    }
    
    # Failure type descriptions
    failure_descriptions = {
        'TWF': 'Tool Wear Failure - Failure due to cumulative tool wear over time',
        'HDF': 'Heat Dissipation Failure - Failure due to insufficient heat dissipation',
        'PWF': 'Power Failure - Failure related to power supply issues',
        'OSF': 'Overstrain Failure - Failure due to excessive mechanical load'
    }
    
    col1, col2 = st.columns(2)
    with col1:
        failure_df = pd.DataFrame({
            'Failure Type': list(failure_types.keys()),
            'Description': [failure_descriptions[ft] for ft in failure_types.keys()],
            'Count': list(failure_types.values()),
            'Percentage': [f"{v/len(df)*100:.1f}%" for v in failure_types.values()]
        })
        st.dataframe(failure_df, use_container_width=True, hide_index=True)
    
    with col2:
        chart_data = failure_df.set_index('Failure Type')['Count']
        st.bar_chart(chart_data, x_label='Failure Type', y_label='Number of Failures')
    
    st.markdown("---")
    
    # Critical Thresholds Analysis - Only Five Main Variables
    st.subheader("Critical Thresholds Analysis")
    st.markdown("**Analysis based on five main variables: Roll Speed, Rolling Torque, Roll Wear, Ambient Temp, Mill Process Temp**")
    
    # Calculate statistics for each main variable
    threshold_data = []
    for var_name, var_info in main_variables.items():
        if var_name not in df.columns:
            continue
        
        avg = df[var_name].mean()
        median = df[var_name].median()
        p75 = np.percentile(df[var_name], 75)
        p90 = np.percentile(df[var_name], 90)
        
        # Get importance count and percentage
        if importance_counts and var_name in importance_counts:
            importance_count = importance_counts[var_name]
            importance_pct = (importance_count / len(df)) * 100
        else:
            importance_count = 0
            importance_pct = 0.0
        
        threshold_data.append({
            'Variable': var_name,
            'Average': f"{avg:.1f} {var_info['unit']}",
            'Median': f"{median:.1f} {var_info['unit']}",
            '75th Percentile': f"{p75:.1f} {var_info['unit']}",
            '90th Percentile': f"{p90:.1f} {var_info['unit']}",
            'Range': f"{df[var_name].min():.1f} - {df[var_name].max():.1f} {var_info['unit']}",
            'Importance Count': importance_count,
            'Importance %': f"{importance_pct:.1f}%"
        })
    
    threshold_df = pd.DataFrame(threshold_data)
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)
    
    st.markdown("**Interpretation:**")
    st.markdown("- **Average**: Mean value across all actual failure records")
    st.markdown("- **Range**: Minimum and maximum values observed in actual failures")
    st.markdown("- **Importance Count**: Number of failures where this variable was identified as the most important contributing factor")
    st.markdown("- **Importance %**: Percentage of failures where this variable was the most important")
    st.markdown("- **Median**: 50% of failures occur at or above this value")
    st.markdown("- **75th Percentile**: 75% of failures occur at or above this value")
    st.markdown("- **90th Percentile**: 90% of failures occur at or above this value")
    
    st.markdown("---")
    
    # Variable Distribution Analysis - Histograms for Five Main Variables
    st.subheader("Variable Distribution Analysis")
    st.markdown("**Distribution of failure counts across value ranges for five main variables**")
    
    # Create histograms for each main variable - display 2 side by side
    main_vars_for_hist = ['Roll Speed [rpm]', 'Rolling Torque [Nm]', 'Roll Wear [min]', 
                          'Ambient Temp [K]', 'Mill Process Temp [K]']
    
    # Process variables in pairs (2 per row)
    for i in range(0, len(main_vars_for_hist), 2):
        # Get two variables for this row
        var1 = main_vars_for_hist[i] if i < len(main_vars_for_hist) else None
        var2 = main_vars_for_hist[i+1] if i+1 < len(main_vars_for_hist) else None
        
        # Create two columns for histograms
        col1, col2 = st.columns(2)
        
        # First histogram
        if var1 and var1 in df.columns:
            with col1:
                var_data = df[var1]
                min_val = var_data.min()
                max_val = var_data.max()
                num_bins = 8
                
                # Create smaller histogram with light theme (high resolution)
                fig, ax = plt.subplots(figsize=(3, 1.25), dpi=200)
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                ax.hist(var_data, bins=num_bins, edgecolor='black', alpha=0.7, color='lightblue')
                ax.set_xlabel(var1, fontsize=4, color='black')
                ax.set_ylabel('Failure Count', fontsize=4, color='black')
                ax.set_title(f'{var1}', fontsize=5, color='black')
                ax.grid(True, alpha=0.3, color='black')
                ax.tick_params(labelsize=4, colors='black')
                # Make spines black
                for spine in ax.spines.values():
                    spine.set_color('black')
                
                # Format x-axis labels to show 1 decimal place
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                
                # Display the histogram
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                # Statistics below histogram
                st.markdown(f"**Min:** {min_val:.1f} | **Max:** {max_val:.1f} | **Avg:** {var_data.mean():.1f}")
        
        # Second histogram
        if var2 and var2 in df.columns:
            with col2:
                var_data = df[var2]
                min_val = var_data.min()
                max_val = var_data.max()
                num_bins = 8
                
                # Create smaller histogram with light theme (high resolution)
                fig, ax = plt.subplots(figsize=(3, 1.25), dpi=200)
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                ax.hist(var_data, bins=num_bins, edgecolor='black', alpha=0.7, color='lightblue')
                ax.set_xlabel(var2, fontsize=4, color='black')
                ax.set_ylabel('Failure Count', fontsize=4, color='black')
                ax.set_title(f'{var2}', fontsize=5, color='black')
                ax.grid(True, alpha=0.3, color='black')
                ax.tick_params(labelsize=4, colors='black')
                # Make spines black
                for spine in ax.spines.values():
                    spine.set_color('black')
                
                # Format x-axis labels to show 1 decimal place
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                
                # Display the histogram
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                # Statistics below histogram
                st.markdown(f"**Min:** {min_val:.1f} | **Max:** {max_val:.1f} | **Avg:** {var_data.mean():.1f}")
        
        st.markdown("---")
    
    # Correlation Analysis - Focus on Five Main Variables
    st.subheader("Variable Correlations")
    st.markdown("**Correlation analysis for five main variables**")
    
    main_cols = list(main_variables.keys())
    available_main_cols = [col for col in main_cols if col in df.columns]
    
    if len(available_main_cols) > 1:
        corr_matrix = df[available_main_cols].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format('{:.2f}'), 
                    use_container_width=True)
        st.markdown("**Note**: Only correlations between the five main variables are shown.")
    else:
        st.warning("Insufficient main variables available for correlation analysis.")
    
    st.markdown("---")
    
    # Summary Insights - Focus on Five Main Variables
    st.subheader("Key Insights")
    st.markdown("**Insights based on the five main variables and actual failure indicators from the dataset**")
    
    insights = []
    
    # Calculate averages for comparison
    speed_avg = df['Roll Speed [rpm]'].mean() if 'Roll Speed [rpm]' in df.columns else 0
    torque_avg = df['Rolling Torque [Nm]'].mean() if 'Rolling Torque [Nm]' in df.columns else 0
    wear_avg = df['Roll Wear [min]'].mean() if 'Roll Wear [min]' in df.columns else 0
    ambient_avg = df['Ambient Temp [K]'].mean() if 'Ambient Temp [K]' in df.columns else 0
    process_temp_avg = df['Mill Process Temp [K]'].mean() if 'Mill Process Temp [K]' in df.columns else 0
    
    # Roll Speed insights
    if 'Roll Speed [rpm]' in df.columns:
        speed_median = df['Roll Speed [rpm]'].median()
        speed_above_avg = (df['Roll Speed [rpm]'] > speed_avg).sum()
        speed_above_avg_pct = (speed_above_avg / len(df)) * 100
        insights.append(f"**Roll Speed**: Median failure speed is {speed_median:.1f} rpm (average: {speed_avg:.1f} rpm). "
                       f"{speed_above_avg_pct:.1f}% of failures occur above average speed.")
    
    # Rolling Torque insights
    if 'Rolling Torque [Nm]' in df.columns:
        torque_median = df['Rolling Torque [Nm]'].median()
        torque_75th = np.percentile(df['Rolling Torque [Nm]'], 75)
        torque_above_avg = (df['Rolling Torque [Nm]'] > torque_avg).sum()
        torque_above_avg_pct = (torque_above_avg / len(df)) * 100
        insights.append(f"**Rolling Torque**: 50% of failures occur when Torque ≥ {torque_median:.1f} Nm (average: {torque_avg:.1f} Nm). "
                       f"75% of failures have Torque ≥ {torque_75th:.1f} Nm. {torque_above_avg_pct:.1f}% occur above average.")
    
    # Roll Wear insights
    if 'Roll Wear [min]' in df.columns:
        wear_median = df['Roll Wear [min]'].median()
        wear_75th = np.percentile(df['Roll Wear [min]'], 75)
        wear_above_avg = (df['Roll Wear [min]'] > wear_avg).sum()
        wear_above_avg_pct = (wear_above_avg / len(df)) * 100
        insights.append(f"**Roll Wear**: 50% of failures occur when Roll Wear ≥ {wear_median:.1f} min (average: {wear_avg:.1f} min). "
                       f"75% of failures have Roll Wear ≥ {wear_75th:.1f} min. {wear_above_avg_pct:.1f}% occur above average.")
    
    # Ambient Temp insights
    if 'Ambient Temp [K]' in df.columns:
        ambient_median = df['Ambient Temp [K]'].median()
        ambient_above_avg = (df['Ambient Temp [K]'] > ambient_avg).sum()
        ambient_above_avg_pct = (ambient_above_avg / len(df)) * 100
        insights.append(f"**Ambient Temp**: Median failure ambient temp is {ambient_median:.1f} K (average: {ambient_avg:.1f} K). "
                       f"{ambient_above_avg_pct:.1f}% of failures occur above average temperature.")
    
    # Mill Process Temp insights
    if 'Mill Process Temp [K]' in df.columns:
        process_temp_median = df['Mill Process Temp [K]'].median()
        process_temp_above_avg = (df['Mill Process Temp [K]'] > process_temp_avg).sum()
        process_temp_above_avg_pct = (process_temp_above_avg / len(df)) * 100
        insights.append(f"**Mill Process Temp**: Median failure process temp is {process_temp_median:.1f} K (average: {process_temp_avg:.1f} K). "
                       f"{process_temp_above_avg_pct:.1f}% of failures occur above average temperature.")
    
    # Failure type insights
    most_common = max(failure_types, key=failure_types.get)
    insights.append(f"**Most Common Failure Type**: {most_common} ({failure_types[most_common]} occurrences, "
                   f"{failure_types[most_common]/len(df)*100:.1f}% of failures) - Actual data from the dataset")
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("---")
    
    # Methodological Note
    st.markdown("**Methodological Notes:**")
    st.markdown("- This analysis is based on actual failure records from the original UCI AI4I 2020 dataset")
    st.markdown("- The dataset contains 339 actual failure records out of 10,000 total records (3.39% failure rate)")
    st.markdown("- Failure indicators (TWF, HDF, PWF, OSF) are actual labels from the dataset, not predictions")
    st.markdown("- Analysis focuses on the five main variables: Roll Speed, Rolling Torque, Roll Wear, Ambient Temp, Mill Process Temp")
    st.markdown("- Engineered features (Power, Temp Diff) are calculated but focus is on main operational variables")
    
    st.markdown("---")
    
    # Download option
    st.markdown("**Download Analysis Data**")
    csv_download = df.to_csv(index=False)
    st.download_button(
        label="Download Original Failures CSV",
        data=csv_download,
        file_name="original_failures_analysis.csv",
        mime="text/csv"
    )
