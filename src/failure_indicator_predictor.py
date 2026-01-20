"""
Failure Indicator Prediction Module for RollingSense Predictive Maintenance System.
Trains logistic regression models to predict failure type indicators (TWF, HDF, PWF, OSF, RNF)
based on operational parameters and engineered features.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config


class FailureIndicatorPredictor:
    """
    Trains and manages models to predict failure type indicators.
    Each indicator (TWF, HDF, PWF, OSF, RNF) has its own logistic regression model.
    """
    
    def __init__(self):
        """Initialize the failure indicator predictor."""
        self.models = {}
        self.cv_scores = {}
        
    def train_indicators(self, X, y_indicators, cv_folds=10):
        """
        Train logistic regression models for each failure indicator.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (same features used for main failure prediction)
        y_indicators : pd.DataFrame
            DataFrame with columns: TWF, HDF, PWF, OSF, RNF
        cv_folds : int
            Number of CV folds for evaluation (default: 10)
        
        Returns:
        --------
        dict : CV scores for each indicator model
        """
        print("\nTraining failure indicator prediction models...")
        indicator_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Create Stratified K-Fold for evaluation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
        
        for indicator in indicator_names:
            if indicator not in y_indicators.columns:
                print(f"  Warning: {indicator} not found in data, skipping...")
                continue
            
            y_indicator = y_indicators[indicator].values
            
            # Check if there are both classes
            if len(np.unique(y_indicator)) < 2:
                print(f"  Warning: {indicator} has only one class, skipping...")
                continue
            
            print(f"  Training {indicator} predictor...")
            
            # Train logistic regression model
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y_indicator)
            
            # Evaluate using cross-validation
            cv_scores = cross_val_score(model, X, y_indicator, cv=skf, scoring='f1_macro')
            
            self.models[indicator] = model
            self.cv_scores[indicator] = {
                'mean': float(np.mean(cv_scores)),
                'std': float(np.std(cv_scores))
            }
            
            print(f"    CV F1-Score: {self.cv_scores[indicator]['mean']:.4f} "
                  f"(±{self.cv_scores[indicator]['std']:.4f})")
        
        print(f"Trained {len(self.models)} indicator prediction models.")
        return self.cv_scores
    
    def predict(self, X):
        """
        Predict failure indicators for given feature matrix.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        
        Returns:
        --------
        dict : Dictionary with predictions for each indicator
        """
        predictions = {}
        for indicator, model in self.models.items():
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            probability = proba[1] if len(proba) > 1 else proba[0]
            
            predictions[indicator] = {
                'value': int(pred),
                'probability': float(probability)
            }
        
        return predictions
    
    def save(self, file_path):
        """
        Save the trained indicator prediction models.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to save the models
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'cv_scores': self.cv_scores
            }, f, protocol=4)  # Protocol 4 for Python 3.8+ compatibility
        print(f"Failure indicator models saved to {file_path}")
    
    def load(self, file_path):
        """
        Load trained indicator prediction models.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to load the models from
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.cv_scores = data.get('cv_scores', {})
            print(f"Failure indicator models loaded from {file_path}")
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as e:
            raise RuntimeError(f"Error loading failure indicator models. This may be due to Python version incompatibility. Original error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading failure indicator models: {str(e)}")


def train_failure_indicators(X, df_original, preprocessor, cv_folds=10):
    """
    Train failure indicator prediction models using the same features as main model.
    
    Parameters:
    -----------
    X : np.ndarray
        Transformed feature matrix (after preprocessing)
    df_original : pd.DataFrame
        Original dataframe with failure indicator columns (TWF, HDF, PWF, OSF, RNF)
    preprocessor : DataPreprocessor
        Preprocessor used for main model (to ensure same features)
    cv_folds : int
        Number of CV folds (default: 10)
    
    Returns:
    --------
    FailureIndicatorPredictor : Trained predictor
    """
    predictor = FailureIndicatorPredictor()
    
    # Extract indicator columns from original dataframe
    indicator_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    y_indicators = df_original[indicator_columns].copy()
    
    # Train models
    predictor.train_indicators(X, y_indicators, cv_folds=cv_folds)
    
    return predictor
