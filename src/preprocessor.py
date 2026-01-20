"""
Data Preprocessing Module for RollingSense Predictive Maintenance System.
Handles data loading, column renaming, feature engineering, encoding, and scaling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

import config


class DataPreprocessor:
    """
    Handles all data preprocessing steps for the RollingSense project.
    """
    
    def __init__(self):
        """Initialize the preprocessor with transformers."""
        self.column_transformer = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.correlation_info = None
        self.columns_to_drop = []
        
    def load_data(self, file_path=None):
        """
        Load the UCI AI4I 2020 dataset.
        
        Parameters:
        -----------
        file_path : str or Path, optional
            Path to the CSV file. If None, downloads from UCI repository.
        
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        if file_path is None:
            file_path = config.RAW_DATA_DIR / "ai4i2020.csv"
            
        # Check if file exists, if not, download it
        if not Path(file_path).exists():
            print(f"Dataset not found at {file_path}. Downloading from UCI repository...")
            df = pd.read_csv(config.DATASET_URL)
            config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
        else:
            df = pd.read_csv(file_path)
        
        return df
    
    def rename_columns(self, df):
        """
        Rename columns to rolling mill context.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame : DataFrame with renamed columns
        """
        rename_map = {
            'Rotational speed [rpm]': 'Roll Speed [rpm]',
            'Torque [Nm]': 'Rolling Torque [Nm]',
            'Tool wear [min]': 'Roll Wear [min]',
            'Air temperature [K]': 'Ambient Temp [K]',
            'Process temperature [K]': 'Mill Process Temp [K]'
        }
        
        df_renamed = df.rename(columns=rename_map)
        return df_renamed
    
    def engineer_features(self, df):
        """
        Create domain-knowledge based features:
        - Power [W]: Rolling Torque * (Roll Speed * 2 * PI / 60)
        - Temp Difference [K]: Mill Process Temp - Ambient Temp
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame : DataFrame with new features
        """
        df = df.copy()
        
        # Calculate Power [W] = Torque * Angular Velocity
        # Angular velocity = (Roll Speed * 2 * PI) / 60 (converting rpm to rad/s)
        df['Power [W]'] = df['Rolling Torque [Nm]'] * (df['Roll Speed [rpm]'] * 2 * np.pi / 60)
        
        # Calculate Temperature Difference
        df['Temp Difference [K]'] = df['Mill Process Temp [K]'] - df['Ambient Temp [K]']
        
        return df
    
    def check_correlation(self, X, threshold=0.90):
        """
        Check for high correlations between numeric features and identify columns to drop.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        threshold : float
            Correlation threshold (default: 0.90)
        
        Returns:
        --------
        dict : Information about correlations and dropped columns
        """
        # Get only numeric columns for correlation check
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            return {
                'high_corr_pairs': [],
                'columns_to_drop': [],
                'message': 'No numeric columns to check for correlation.'
            }
        
        # Calculate correlation matrix
        corr_matrix = numeric_X.corr().abs()
        
        # Find high correlation pairs
        high_corr_pairs = []
        columns_to_drop = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if corr_value >= threshold:
                    high_corr_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': float(corr_value)
                    })
                    # Drop the second feature (arbitrary choice, could be improved)
                    columns_to_drop.add(col2)
        
        self.columns_to_drop = list(columns_to_drop)
        
        correlation_info = {
            'high_corr_pairs': high_corr_pairs,
            'columns_to_drop': self.columns_to_drop,
            'threshold': threshold,
            'message': f'Found {len(high_corr_pairs)} high correlation pairs (>= {threshold}). Dropped {len(columns_to_drop)} features.'
        }
        
        self.correlation_info = correlation_info
        return correlation_info
    
    def prepare_features_and_target(self, df):
        """
        Separate features and target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        X : pd.DataFrame : Feature matrix
        y : pd.Series : Target variable
        """
        # Target is 'Machine failure'
        y = df['Machine failure'].copy()
        
        # Drop non-feature columns and failure type indicators (to avoid data leakage)
        # TWF, HDF, PWF, OSF, RNF are excluded as they are components of the target variable
        columns_to_drop = ['Machine failure', 'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Check for high correlations and drop highly correlated features
        correlation_info = self.check_correlation(X, threshold=0.90)
        if self.columns_to_drop:
            print(f"\nCorrelation Check:")
            print(f"  Threshold: 0.90")
            print(f"  High correlation pairs found: {len(correlation_info['high_corr_pairs'])}")
            for pair in correlation_info['high_corr_pairs']:
                print(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
            print(f"  Dropping columns: {self.columns_to_drop}")
            X = X.drop(columns=self.columns_to_drop)
        else:
            print(f"\nCorrelation Check:")
            print(f"  Threshold: 0.90")
            print(f"  No high correlations found (>= 0.90). All features retained.")
        
        return X, y
    
    def fit_transform(self, X, y=None):
        """
        Fit preprocessor on training data and transform.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target variable (not used, kept for sklearn compatibility)
        
        Returns:
        --------
        X_transformed : np.ndarray : Transformed feature matrix
        """
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remove 'Type' from numeric if present, add to categorical
        if 'Type' in numeric_cols:
            numeric_cols.remove('Type')
        if 'Type' not in categorical_cols and 'Type' in X.columns:
            categorical_cols.append('Type')
        
        # Create transformers
        transformers = []
        
        # One-hot encode categorical columns (Type)
        if categorical_cols:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            )
        
        # Standard scale numeric columns
        if numeric_cols:
            transformers.append(
                ('num', StandardScaler(), numeric_cols)
            )
        
        # Create column transformer
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        
        # Fit and transform
        X_transformed = self.column_transformer.fit_transform(X)
        
        # Store feature names
        self.feature_names = self.column_transformer.get_feature_names_out()
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        X_transformed : np.ndarray : Transformed feature matrix
        """
        if self.column_transformer is None:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        return self.column_transformer.transform(X)
    
    def get_feature_names(self):
        """
        Get feature names after transformation.
        
        Returns:
        --------
        list : Feature names
        """
        if self.feature_names is None:
            raise ValueError("Preprocessor must be fitted first.")
        return self.feature_names.tolist()
    
    def save(self, file_path):
        """
        Save the fitted preprocessor to disk.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to save the preprocessor
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'column_transformer': self.column_transformer,
                'feature_names': self.feature_names,
                'correlation_info': self.correlation_info,
                'columns_to_drop': self.columns_to_drop
            }, f, protocol=4)  # Protocol 4 for Python 3.8+ compatibility
    
    def load(self, file_path):
        """
        Load a fitted preprocessor from disk.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to load the preprocessor from
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.column_transformer = data['column_transformer']
                self.feature_names = data['feature_names']
                self.correlation_info = data.get('correlation_info', None)
                self.columns_to_drop = data.get('columns_to_drop', [])
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as e:
            raise RuntimeError(f"Error loading preprocessor. This may be due to Python version incompatibility. Original error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading preprocessor: {str(e)}")


def preprocess_pipeline(data_path=None):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to the dataset CSV file
    
    Returns:
    --------
    preprocessor : DataPreprocessor : Fitted preprocessor
    X : np.ndarray : Preprocessed features
    y : pd.Series : Target variable
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    print("Loading data...")
    df = preprocessor.load_data(data_path)
    
    # Rename columns
    print("Renaming columns to rolling mill context...")
    df = preprocessor.rename_columns(df)
    
    # Engineer features
    print("Engineering domain-knowledge features...")
    df = preprocessor.engineer_features(df)
    
    # Prepare features and target
    print("Preparing features and target...")
    X, y = preprocessor.prepare_features_and_target(df)
    
    # Fit and transform
    print("Fitting preprocessor and transforming data...")
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"Preprocessing complete. Shape: {X_transformed.shape}")
    print(f"Features: {len(preprocessor.get_feature_names())}")
    
    return preprocessor, X_transformed, y

