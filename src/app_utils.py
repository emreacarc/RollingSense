"""
Utility functions for Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

import config
from src.preprocessor import DataPreprocessor
from src.failure_indicator_predictor import FailureIndicatorPredictor


@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor (cached for performance)."""
    try:
        with open(config.BEST_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        preprocessor_path = config.MODELS_DIR / "preprocessor.pkl"
        preprocessor = DataPreprocessor()
        preprocessor.load(preprocessor_path)
        
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"Model or preprocessor not found: {e}. Please train the model first by running: python train.py")
        return None, None
    except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError, RuntimeError) as e:
        st.error(f"Error loading model files. This may be due to Python version incompatibility.")
        st.error(f"Error details: {str(e)}")
        st.info("💡 **Solution:** Please re-train the models locally with a compatible Python version (3.8-3.11 recommended), then push the updated model files to GitHub.")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error loading models: {str(e)}")
        return None, None


def load_model_report():
    """
    Load the model evaluation report.
    Note: Cache removed to ensure latest data is always displayed.
    """
    try:
        file_path = config.MODEL_REPORT_PATH
        if file_path.exists():
            with open(file_path, 'r') as f:
                report = json.load(f)
            # Validate report has expected structure
            if 'models' not in report:
                st.warning("Model report format is invalid. Please re-run training: `python train.py`")
                return None
            # Check if we have all 5 models
            expected_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
            report_models = list(report['models'].keys())
            if len(report_models) < 5:
                st.warning(f"Model report may be outdated. Found {len(report_models)} models, expected 5. Please re-run training: `python train.py`")
            return report
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model report: {e}")
        return None


def predict_with_model(model, preprocessor, input_dict):
    """
    Make prediction using the trained model.
    Note: Failure type indicators (TWF, HDF, PWF, OSF, RNF) are not used as features
    to avoid data leakage, as they are components of the target variable.
    """
    # Calculate engineered features
    power = input_dict['Rolling Torque [Nm]'] * (input_dict['Roll Speed [rpm]'] * 2 * np.pi / 60)
    temp_diff = input_dict['Mill Process Temp [K]'] - input_dict['Ambient Temp [K]']
    
    # Build base DataFrame (excluding failure type indicators)
    base_features = {
        'Type': [input_dict['Type']],
        'Ambient Temp [K]': [input_dict['Ambient Temp [K]']],
        'Mill Process Temp [K]': [input_dict['Mill Process Temp [K]']],
        'Roll Speed [rpm]': [input_dict['Roll Speed [rpm]']],
        'Rolling Torque [Nm]': [input_dict['Rolling Torque [Nm]']],
        'Roll Wear [min]': [input_dict['Roll Wear [min]']],
        'Temp Difference [K]': [temp_diff]
    }
    
    # Add Power [W] only if it wasn't dropped during training
    if not (preprocessor.columns_to_drop and 'Power [W]' in preprocessor.columns_to_drop):
        base_features['Power [W]'] = [power]
    
    df_input = pd.DataFrame(base_features)
    
    # Drop columns removed during training due to high correlation
    if preprocessor.columns_to_drop:
        df_input = df_input.drop(columns=[col for col in preprocessor.columns_to_drop if col in df_input.columns])
    
    # Transform and predict
    X_transformed = preprocessor.transform(df_input)
    prediction = model.predict(X_transformed)[0]
    
    # Get probability
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_transformed)[0]
        probability = proba[1] if len(proba) > 1 else proba[0]
    else:
        probability = float(prediction)
    
    return int(prediction), float(probability)


def load_failure_indicator_predictor():
    """Load the failure indicator predictor models."""
    try:
        predictor = FailureIndicatorPredictor()
        predictor.load(config.FAILURE_INDICATOR_PREDICTOR_PATH)
        return predictor
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


def predict_failure_indicators(predictor, preprocessor, input_dict):
    """
    Predict failure type indicators (TWF, HDF, PWF, OSF, RNF) for given input.
    
    Parameters:
    -----------
    predictor : FailureIndicatorPredictor
        Trained indicator predictor
    preprocessor : DataPreprocessor
        Preprocessor for feature transformation
    input_dict : dict
        Input dictionary with operational parameters
    
    Returns:
    --------
    dict : Dictionary with predicted indicator values (0 or 1) for each type
    """
    if predictor is None:
        # Fallback: return zeros if predictor not available
        return {'TWF': 0, 'HDF': 0, 'PWF': 0, 'OSF': 0, 'RNF': 0}
    
    # Calculate engineered features
    power = input_dict['Rolling Torque [Nm]'] * (input_dict['Roll Speed [rpm]'] * 2 * np.pi / 60)
    temp_diff = input_dict['Mill Process Temp [K]'] - input_dict['Ambient Temp [K]']
    
    # Build base DataFrame (same as main prediction)
    base_features = {
        'Type': [input_dict['Type']],
        'Ambient Temp [K]': [input_dict['Ambient Temp [K]']],
        'Mill Process Temp [K]': [input_dict['Mill Process Temp [K]']],
        'Roll Speed [rpm]': [input_dict['Roll Speed [rpm]']],
        'Rolling Torque [Nm]': [input_dict['Rolling Torque [Nm]']],
        'Roll Wear [min]': [input_dict['Roll Wear [min]']],
        'Temp Difference [K]': [temp_diff]
    }
    
    # Add Power [W] only if it wasn't dropped during training
    if not (preprocessor.columns_to_drop and 'Power [W]' in preprocessor.columns_to_drop):
        base_features['Power [W]'] = [power]
    
    df_input = pd.DataFrame(base_features)
    
    # Drop columns removed during training due to high correlation
    if preprocessor.columns_to_drop:
        df_input = df_input.drop(columns=[col for col in preprocessor.columns_to_drop if col in df_input.columns])
    
    # Transform features
    X_transformed = preprocessor.transform(df_input)
    
    # Predict indicators
    indicator_predictions = predictor.predict(X_transformed)
    
    # Extract values
    result = {}
    for indicator in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
        if indicator in indicator_predictions:
            result[indicator] = indicator_predictions[indicator]['value']
        else:
            result[indicator] = 0
    
    return result
