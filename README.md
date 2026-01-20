# RollingSense - Predictive Maintenance System

A production-grade Predictive Maintenance system for Rolling Mills, simulating real-world industrial scenarios. This system leverages advanced machine learning techniques to predict equipment failures before they occur, enabling proactive maintenance and reducing costly downtime.

## Project Overview

**RollingSense** is designed specifically for Rolling Mills, simulating scenarios similar to those encountered in steel production facilities such as Outokumpu. The system monitors critical parameters and predicts machine failures based on sensor readings and operational parameters.

### Key Features

- **Advanced Feature Engineering**: Domain-knowledge based features (Power, Temperature Difference)
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Rigorous Evaluation**: 10-Fold Stratified Cross-Validation
- **Smart Model Selection**: Balances F1-Score and inference speed
- **Real-time Monitoring**: Streamlit dashboard for live predictions
- **Production-Grade**: Modular code structure, comprehensive reporting

## Project Structure

```
RollingSense/
├── config.py              # Configuration and paths
├── train.py               # Training script
├── app.py                 # Streamlit dashboard (main entry point)
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── preprocessor.py   # Data preprocessing module
│   ├── model_trainer.py  # Model training and evaluation
│   ├── app_ui.py         # Streamlit UI components
│   ├── app_utils.py      # Utility functions for app
│   └── failure_indicator_predictor.py  # Failure indicator prediction
├── models/               # Saved models and reports (gitignored)
├── data/
│   ├── raw/             # Raw dataset (gitignored)
│   └── processed/       # Processed data (gitignored)
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/emreacarc/RollingSense.git
   cd RollingSense
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Models

Run the training pipeline to download data, preprocess it, train models, and select the best one:

```bash
python train.py
```

This will:
- Download the UCI AI4I 2020 dataset (if not present)
- Preprocess data with feature engineering
- Train 5 models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- Evaluate models using 10-Fold Cross-Validation
- Measure inference speed for each model
- Select the best model based on F1-Score and speed
- Save the best model and evaluation report

**Outputs:**
- `models/best_model.pkl` - Trained best model
- `models/preprocessor.pkl` - Fitted preprocessor
- `models/model_report.json` - Detailed evaluation report

### 2. Run the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser with four main modules:

#### Module 1: Live Monitor
- Input sensor values (Type, Roll Speed, Torque, Wear, Temperatures)
- Real-time failure predictions
- Visual feedback (Green/Red) with probability scores

#### Module 2: Sample Failure Scenarios
- View 10 randomly selected failure scenarios from generated dataset
- See predicted failure indicators and most important contributing features
- Generate new random scenarios from existing dataset

#### Module 3: Failure Insights & Analytics
- Overview statistics from actual failure data
- Failure type distribution
- Critical thresholds analysis
- Variable correlations
- Variable distribution histograms
- Key insights and methodological notes

#### Module 4: About Project
- Project context and technologies
- Feature Engineering: Explanation of Power and Temp Difference features
- Model Evaluation: Comparison table of all models
- Champion Model: Details about the selected best model

## Feature Engineering

The system includes domain-knowledge based feature engineering:

### 1. Power [W]
- **Formula:** `Rolling Torque [Nm] × (Roll Speed [rpm] × 2π / 60)`
- **Rationale:** Captures mechanical work and system load. Higher power indicates increased stress on components.

### 2. Temp Difference [K]
- **Formula:** `Mill Process Temp [K] - Ambient Temp [K]`
- **Rationale:** Indicates heat generation during operation. Abnormal thermal conditions may precede failures.

### Important Note: Failure Type Indicators
The failure type indicators (TWF, HDF, PWF, OSF, RNF) are **not used as features** in model training to avoid data leakage, as they are components of the target variable. The model predicts failures based solely on operational parameters (Type, Roll Speed, Rolling Torque, Roll Wear, Temperatures) and engineered features (Power, Temp Difference).

## Model Selection Logic

The system uses a smart selection algorithm that balances predictive performance and inference speed:

1. **Evaluate all models** using 10-Fold Cross-Validation (Accuracy and F1-Score)
2. **Measure inference speed** for 10,000 samples
3. **Select best model:**
   - If F1 difference between top 2 models < 1%: Choose the faster model
   - Otherwise: Choose the model with highest F1-Score

This ensures both high accuracy and acceptable inference speed for production environments.

### Model Evaluation Results

The following table shows the performance of all evaluated models:

| Model | CV Accuracy | CV F1-Score (Macro) | Inference Time (ms) |
|-------|-------------|---------------------|---------------------|
| **LightGBM** | **0.9884** | **0.8951** | **9.14** |
| CatBoost | 0.9873 | 0.8834 | 49.63 |
| XGBoost | 0.9865 | 0.8816 | 3.99 |
| Random Forest | 0.9862 | 0.8762 | 62.76 |
| Logistic Regression | 0.9696 | 0.6381 | 0.89 |

### Selected Model: LightGBM

**Why LightGBM was selected:**
- **Highest F1-Score**: LightGBM achieved the highest F1-Score (0.8951), indicating the best balance between precision and recall across all failure types
- **Excellent Accuracy**: Second-highest accuracy (0.9884) among all models
- **Reasonable Inference Speed**: While not the fastest (9.14ms), the inference speed is acceptable for real-time predictions in production environments
- **Decision Rationale**: The F1-Score difference between LightGBM (0.8951) and the second-best model, CatBoost (0.8834), is 0.0117 (1.17%), which exceeds the 1% threshold. Therefore, the model with the highest F1-Score was selected according to our selection algorithm

**Model Performance Highlights:**
- **LightGBM** outperforms other models with a 1.17% higher F1-Score than CatBoost
- Gradient boosting models (LightGBM, CatBoost, XGBoost) significantly outperform the baseline Logistic Regression model
- All ensemble methods show strong predictive performance, with accuracy above 98.5%

## Technical Details

### Models Trained and Evaluated
- **Logistic Regression**: Linear baseline model - Provides a simple, interpretable baseline (F1: 0.6381, Speed: 0.89ms)
- **Random Forest**: Ensemble of decision trees - Robust model with good performance (F1: 0.8762, Speed: 62.76ms)
- **XGBoost**: Gradient boosting classifier - Fast and accurate gradient boosting (F1: 0.8816, Speed: 3.99ms)
- **LightGBM**: Gradient boosting with leaf-wise tree growth - **Selected champion model** (F1: 0.8951, Speed: 9.14ms)
- **CatBoost**: Gradient boosting optimized for categorical features - Strong competitor with good performance (F1: 0.8834, Speed: 49.63ms)

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score (Macro)**: Balanced measure of precision and recall

### Cross-Validation
- **Method**: Stratified 10-Fold Cross-Validation
- **Purpose**: Robust evaluation, ensuring all folds have similar class distribution

## Configuration

All paths and settings are centralized in `config.py`:

- Dataset URL: UCI AI4I 2020 repository
- CV Folds: 10 (configurable)
- Inference test size: 10,000 samples
- F1 difference threshold: 0.01 (1%)

## License

This project is for portfolio purposes. The dataset used is from the UCI Machine Learning Repository (AI4I 2020 dataset).

## References

- UCI AI4I 2020 Dataset: [https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

## Deployment

### GitHub

1. **Initialize Git repository (if not already done):**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create a new repository on GitHub** and push:
   ```bash
   git remote add origin https://github.com/emreacarc/RollingSense.git
   git branch -M main
   git push -u origin main
   ```

**Note:** Model files (`*.pkl`), data files (`*.csv`), and cache directories are excluded via `.gitignore` to keep the repository size manageable. You'll need to train the models locally or provide them separately.

### Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

2. **Sign in with your GitHub account**

3. **Click "New app" and select:**
   - Repository: `emreacarc/RollingSense`
   - Branch: `main`
   - Main file: `app.py`

4. **Important:** Before deploying, ensure you have:
   - Trained models available (either commit them or use Streamlit Secrets for model storage)
   - The `models/` directory structure exists (even if empty)
   - All dependencies listed in `requirements.txt`

5. **For models on Streamlit Cloud, you have two options:**
   
   **Option A:** Commit model files (if under 100MB total)
   - Temporarily remove `models/*.pkl` and `models/*.csv` from `.gitignore`
   - Commit and push model files
   - Add them back to `.gitignore` after deployment
   
   **Option B:** Use Streamlit Secrets or external storage
   - Store models in cloud storage (S3, Google Cloud Storage, etc.)
   - Download models at runtime using Streamlit Secrets for credentials
   - Or use GitHub Releases to attach model files

6. **Deploy!** Streamlit Cloud will automatically:
   - Install dependencies from `requirements.txt`
   - Run `streamlit run app.py`
   - Provide you with a public URL

**Note:** If your models are large, consider using Option B or training models directly on Streamlit Cloud by running `train.py` as part of the deployment process (though this may increase deployment time).

## Notes

- All code, comments, and UI text are in English
- No `random_state` parameters are used (reproducibility through other means)
- Modular structure for easy maintenance and extension
- Model and data files are excluded from Git to keep repository size manageable

