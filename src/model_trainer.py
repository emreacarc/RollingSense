"""
Model Training and Evaluation Module for RollingSense Predictive Maintenance System.
Trains multiple models, evaluates using 10-Fold CV, measures inference speed, and selects the best model.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import config


class ModelTrainer:
    """
    Trains multiple models, evaluates them, and selects the best one.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.cv_results = {}
        self.inference_times = {}
        self.selected_model = None
        self.selected_model_name = None
        
    def initialize_models(self):
        """
        Initialize the models to train.
        Note: No random_state is used per requirements.
        """
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbose=False),
            'LightGBM': LGBMClassifier(verbose=-1),
            'CatBoost': CatBoostClassifier(verbose=False, iterations=100)
        }
    
    def train_and_evaluate(self, X, y, cv_folds=10):
        """
        Train models and evaluate using Stratified K-Fold Cross-Validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target variable
        cv_folds : int
            Number of CV folds (default: 10)
        
        Returns:
        --------
        dict : CV results for all models
        """
        print(f"\nStarting {cv_folds}-Fold Cross-Validation evaluation...")
        
        # Initialize models if not done
        if not self.models:
            self.initialize_models()
        
        # Create Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Calculate cross-validation scores
            cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
            
            # Store results
            self.cv_results[model_name] = {
                'CV_Accuracy_Mean': float(np.mean(cv_accuracy)),
                'CV_Accuracy_Std': float(np.std(cv_accuracy)),
                'CV_F1_Score_Mean': float(np.mean(cv_f1)),
                'CV_F1_Score_Std': float(np.std(cv_f1))
            }
            
            print(f"  Accuracy: {self.cv_results[model_name]['CV_Accuracy_Mean']:.4f} "
                  f"(±{self.cv_results[model_name]['CV_Accuracy_Std']:.4f})")
            print(f"  F1-Score: {self.cv_results[model_name]['CV_F1_Score_Mean']:.4f} "
                  f"(±{self.cv_results[model_name]['CV_F1_Score_Std']:.4f})")
        
        return self.cv_results
    
    def measure_inference_speed(self, X, y, test_size=10000):
        """
        Measure inference speed for each model on a sample of data.
        Models are trained on the full dataset before speed measurement.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (used for training and creating test sample)
        y : pd.Series or np.ndarray
            Target variable (used for training)
        test_size : int
            Number of samples to use for speed test (default: 10000)
        
        Returns:
        --------
        dict : Inference times in milliseconds for each model
        """
        print(f"\nMeasuring inference speed on {test_size} samples...")
        
        # First, train all models on full dataset for inference speed testing
        print("  Training models on full dataset for speed measurement...")
        for model_name, model in self.models.items():
            model.fit(X, y)
        
        # Create test sample
        if X.shape[0] < test_size:
            X_test = X
            print(f"  Note: Dataset has {X.shape[0]} samples, using all available.")
        else:
            # Randomly sample without replacement
            indices = np.random.choice(X.shape[0], size=test_size, replace=False)
            X_test = X[indices]
        
        # Measure inference time for each model
        for model_name, model in self.models.items():
            print(f"  Testing {model_name}...")
            
            # Time the prediction
            start_time = time.time()
            _ = model.predict(X_test)
            end_time = time.time()
            
            # Convert to milliseconds
            inference_time_ms = (end_time - start_time) * 1000
            self.inference_times[model_name] = float(inference_time_ms)
            
            print(f"    Inference time: {inference_time_ms:.2f} ms")
        
        return self.inference_times
    
    def select_best_model(self, f1_threshold=0.01):
        """
        Select the best model based on F1-Score and inference speed.
        Logic: If F1 difference < threshold, choose faster model; else choose highest F1.
        
        Parameters:
        -----------
        f1_threshold : float
            Threshold for F1 difference (default: 0.01 = 1%)
        
        Returns:
        --------
        str : Name of the selected best model
        """
        if not self.cv_results or not self.inference_times:
            raise ValueError("Must run train_and_evaluate and measure_inference_speed first.")
        
        # Sort models by F1-Score (descending)
        models_by_f1 = sorted(
            self.cv_results.items(),
            key=lambda x: x[1]['CV_F1_Score_Mean'],
            reverse=True
        )
        
        best_model_name, best_results = models_by_f1[0]
        best_f1 = best_results['CV_F1_Score_Mean']
        
        # If only one model, select it
        if len(models_by_f1) == 1:
            self.selected_model_name = best_model_name
            self.selected_model = self.models[best_model_name]
            return best_model_name
        
        # Get second-best model
        second_best_model_name, second_best_results = models_by_f1[1]
        second_best_f1 = second_best_results['CV_F1_Score_Mean']
        
        # Calculate F1 difference
        f1_difference = abs(best_f1 - second_best_f1)
        
        print(f"\nModel Selection Analysis:")
        print(f"  Best F1: {best_model_name} ({best_f1:.4f})")
        print(f"  Second-best F1: {second_best_model_name} ({second_best_f1:.4f})")
        print(f"  F1 Difference: {f1_difference:.4f} (Threshold: {f1_threshold})")
        
        # Apply selection logic
        if f1_difference < f1_threshold:
            # F1 difference is small, choose faster model between top 2
            best_time = self.inference_times[best_model_name]
            second_best_time = self.inference_times[second_best_model_name]
            
            if best_time < second_best_time:
                selected = best_model_name
                reason = f"F1 difference < {f1_threshold*100}%, selected faster model"
            else:
                selected = second_best_model_name
                reason = f"F1 difference < {f1_threshold*100}%, selected faster model"
            
            print(f"  Selected: {selected} (Reason: {reason})")
        else:
            # F1 difference is significant, choose highest F1
            selected = best_model_name
            reason = f"F1 difference >= {f1_threshold*100}%, selected highest F1-Score"
            print(f"  Selected: {selected} (Reason: {reason})")
        
        self.selected_model_name = selected
        self.selected_model = self.models[selected]
        
        return selected
    
    def train_final_model(self, X, y):
        """
        Train the selected model on the full dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target variable
        """
        if self.selected_model is None:
            raise ValueError("Must select best model first. Call select_best_model().")
        
        print(f"\nTraining final model: {self.selected_model_name}...")
        self.selected_model.fit(X, y)
        print("Training complete.")
    
    def save_model(self, file_path):
        """
        Save the selected model to disk.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to save the model
        """
        if self.selected_model is None:
            raise ValueError("No model selected to save.")
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.selected_model, f, protocol=4)  # Protocol 4 for Python 3.8+ compatibility
        
        print(f"Model saved to {file_path}")
    
    def generate_report(self, output_path):
        """
        Generate a detailed model evaluation report.
        
        Parameters:
        -----------
        output_path : str or Path
            Path to save the report JSON
        """
        if not self.cv_results or not self.inference_times:
            raise ValueError("Must run evaluation first.")
        
        # Build report
        report = {
            'models': {},
            'selected_best_model': self.selected_model_name
        }
        
        # Add results for each model
        for model_name in self.cv_results.keys():
            report['models'][model_name] = {
                'Name': model_name,
                'CV_Accuracy': self.cv_results[model_name]['CV_Accuracy_Mean'],
                'CV_F1_Score': self.cv_results[model_name]['CV_F1_Score_Mean'],
                'Inference_Time_ms': self.inference_times[model_name]
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Model report saved to {output_path}")
        
        return report


def train_models_pipeline(X, y, preprocessor=None, cv_folds=10):
    """
    Complete training and evaluation pipeline.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    preprocessor : DataPreprocessor, optional
        Preprocessor object to get correlation info
    cv_folds : int
        Number of CV folds (default: 10)
    
    Returns:
    --------
    trainer : ModelTrainer : Trained model trainer
    report : dict : Evaluation report
    """
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Train and evaluate
    trainer.train_and_evaluate(X, y, cv_folds=cv_folds)
    
    # Measure inference speed
    trainer.measure_inference_speed(X, y, test_size=config.INFERENCE_TEST_SIZE)
    
    # Select best model
    trainer.select_best_model(f1_threshold=config.F1_DIFFERENCE_THRESHOLD)
    
    # Train final model on full dataset
    trainer.train_final_model(X, y)
    
    # Save model
    trainer.save_model(config.BEST_MODEL_PATH)
    
    # Generate and save report
    report = trainer.generate_report(config.MODEL_REPORT_PATH)
    
    # Add correlation info to report if available
    if preprocessor and preprocessor.correlation_info:
        report['correlation_check'] = preprocessor.correlation_info
    
    # Save updated report
    with open(config.MODEL_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    
    return trainer, report

