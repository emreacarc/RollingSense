"""
Training Script for RollingSense Predictive Maintenance System.
Downloads data, preprocesses it, trains models, and saves the best model.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessor import preprocess_pipeline, DataPreprocessor
from src.model_trainer import train_models_pipeline
from src.failure_indicator_predictor import train_failure_indicators

import config


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("RollingSense - Predictive Maintenance Training Pipeline")
    print("=" * 60)
    
    # Step 1: Preprocess data
    print("\n[Step 1/3] Data Preprocessing")
    print("-" * 60)
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data()
    df = preprocessor.rename_columns(df)
    df = preprocessor.engineer_features(df)
    
    # Get original dataframe for indicator training (before dropping columns)
    df_with_indicators = df.copy()
    
    # Prepare features and target (this will drop indicators)
    X_df, y = preprocessor.prepare_features_and_target(df)
    X_transformed = preprocessor.fit_transform(X_df)
    
    # Save preprocessor for later use
    preprocessor_path = config.MODELS_DIR / "preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Step 2: Train and evaluate main models
    print("\n[Step 2/3] Main Model Training & Evaluation")
    print("-" * 60)
    trainer, report = train_models_pipeline(X_transformed, y, preprocessor=preprocessor, cv_folds=config.CV_FOLDS)
    
    # Step 3: Train failure indicator prediction models
    print("\n[Step 3/3] Failure Indicator Prediction Models")
    print("-" * 60)
    # Use same transformed features as main model
    indicator_predictor = train_failure_indicators(
        X_transformed, 
        df_with_indicators, 
        preprocessor, 
        cv_folds=config.CV_FOLDS
    )
    
    # Save indicator predictor
    indicator_predictor.save(config.FAILURE_INDICATOR_PREDICTOR_PATH)
    print(f"Failure indicator predictor saved to {config.FAILURE_INDICATOR_PREDICTOR_PATH}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Selected Best Model: {report['selected_best_model']}")
    print(f"Main model saved to: {config.BEST_MODEL_PATH}")
    print(f"Failure indicator predictor saved to: {config.FAILURE_INDICATOR_PREDICTOR_PATH}")
    print(f"Report saved to: {config.MODEL_REPORT_PATH}")
    print("\nYou can now run the Streamlit app: streamlit run app.py")


if __name__ == "__main__":
    main()

