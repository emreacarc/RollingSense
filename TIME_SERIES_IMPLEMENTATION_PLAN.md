# Time Series Implementation Plan for RollingSense

## Current Data Structure

- **Total samples**: 10,000
- **Unique Product IDs**: 10,000 (1 row per product)
- **UDI**: Sequential from 1 to 10,000 (can be used as time sequence)
- **Challenge**: No real time series per product (each product has only 1 measurement)

## Implementation Strategy

### Approach: UDI-Based Time Series Simulation

Since each product has only 1 row, we'll treat **UDI as a time sequence** representing the order of operations/measurements in the rolling mill. This allows us to create temporal features based on previous measurements.

---

## Step 1: Add Time Series Feature Engineering to Preprocessor

### New Method: `engineer_temporal_features()`

**Location**: `src/preprocessor.py` → `DataPreprocessor` class

**Features to Add**:

#### A. Lag Features (Previous Values)
- `Roll Speed [rpm]_lag1`, `_lag2`, `_lag3` (previous 1, 2, 3 measurements)
- `Rolling Torque [Nm]_lag1`, `_lag2`
- `Temp Diff [K]_lag1`
- `Roll Wear [min]_lag1` (to calculate wear rate)

#### B. Rolling Window Statistics (Last N measurements)
- `Roll Speed_MA5`, `Roll Speed_Std5` (mean and std of last 5)
- `Rolling Torque_MA5`, `Rolling Torque_Std5`
- `Temp Diff_MA5`, `Temp Diff_Max5`
- `Power_MA5`, `Power_Std5`

#### C. Temporal Patterns
- `Roll Wear_Change_Rate` = (Roll Wear[t] - Roll Wear[t-1]) / 1 (per measurement)
- `Temp_Diff_Trend` = Slope of last 3 temperature differences
- `Speed_Acceleration` = Roll Speed[t] - Roll Speed[t-1]
- `Torque_Change` = Rolling Torque[t] - Rolling Torque[t-1]

#### D. Time-Based Features
- `Measurement_Sequence` = UDI (normalized)
- `Roll_Wear_Age` = Roll Wear / max(Roll Wear) (normalized wear level)

---

## Step 2: Implementation Details

### 2.1 Modify `engineer_features()` Method

```python
def engineer_features(self, df):
    """Create domain-knowledge and temporal features."""
    df = df.copy()
    
    # Sort by UDI to ensure temporal order
    df = df.sort_values('UDI').reset_index(drop=True)
    
    # Existing features
    df['Power [W]'] = df['Rolling Torque [Nm]'] * (df['Roll Speed [rpm]'] * 2 * np.pi / 60)
    df['Temp Difference [K]'] = df['Mill Process Temp [K]'] - df['Ambient Temp [K]']
    
    # NEW: Temporal features
    df = self._add_lag_features(df)
    df = self._add_rolling_statistics(df)
    df = self._add_temporal_patterns(df)
    df = self._add_time_based_features(df)
    
    return df
```

### 2.2 Helper Methods

```python
def _add_lag_features(self, df, lags=[1, 2, 3]):
    """Add lag features for key variables."""
    features_to_lag = [
        'Roll Speed [rpm]',
        'Rolling Torque [Nm]',
        'Temp Difference [K]',
        'Roll Wear [min]'
    ]
    
    for feature in features_to_lag:
        for lag in lags:
            if feature == 'Roll Speed [rpm]' and lag > 3:
                continue  # Only 3 lags for speed
            if feature in ['Rolling Torque [Nm]', 'Temp Difference [K]'] and lag > 1:
                continue  # Only 1 lag for these
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    
    return df

def _add_rolling_statistics(self, df, window=5):
    """Add rolling window statistics."""
    numeric_features = [
        'Roll Speed [rpm]',
        'Rolling Torque [Nm]',
        'Temp Difference [K]',
        'Power [W]'
    ]
    
    for feature in numeric_features:
        df[f'{feature}_MA{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
        df[f'{feature}_Std{window}'] = df[feature].rolling(window=window, min_periods=1).std().fillna(0)
        
        if feature == 'Temp Difference [K]':
            df[f'{feature}_Max{window}'] = df[feature].rolling(window=window, min_periods=1).max()
    
    return df

def _add_temporal_patterns(self, df):
    """Add temporal pattern features."""
    # Roll Wear change rate
    df['Roll Wear_Change_Rate'] = df['Roll Wear [min]'].diff().fillna(0)
    
    # Temperature difference trend (slope of last 3)
    df['Temp_Diff_Trend'] = df['Temp Difference [K]'].rolling(window=3, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    ).fillna(0)
    
    # Speed acceleration
    df['Speed_Acceleration'] = df['Roll Speed [rpm]'].diff().fillna(0)
    
    # Torque change
    df['Torque_Change'] = df['Rolling Torque [Nm]'].diff().fillna(0)
    
    return df

def _add_time_based_features(self, df):
    """Add time-based features."""
    # Normalized measurement sequence
    df['Measurement_Sequence'] = (df['UDI'] - df['UDI'].min()) / (df['UDI'].max() - df['UDI'].min())
    
    # Normalized wear age
    max_wear = df['Roll Wear [min]'].max()
    df['Roll_Wear_Age'] = df['Roll Wear [min]'] / max_wear if max_wear > 0 else 0
    
    return df
```

---

## Step 3: Handle Missing Values

**Issue**: Lag features will have NaN for first rows (no previous values)

**Solution**: Fill NaN with:
- **Forward fill**: Use first available value
- **Or**: Drop first N rows (where N = max lag)
- **Or**: Fill with 0 or mean value

**Recommended**: Forward fill for first rows

```python
# After creating lag features
df = df.fillna(method='ffill').fillna(0)  # Forward fill, then 0 for any remaining
```

---

## Step 4: Update Correlation Check

**Note**: New temporal features may have high correlations. The existing `check_correlation()` method will handle this automatically.

---

## Step 5: Update Prediction Function

**Location**: `src/app_utils.py` → `predict_with_model()`

**Challenge**: For live predictions, we don't have historical data.

**Solution**: 
- For first prediction: Set lag features to 0 or current value
- For subsequent predictions: Store last N predictions and use them as lag features
- Use session state in Streamlit to maintain prediction history

---

## Step 6: Configuration

**Location**: `config.py`

Add time series settings:

```python
# Time Series Settings
TIME_SERIES_ENABLED = True
LAG_FEATURES = [1, 2, 3]  # Lag periods
ROLLING_WINDOW = 5  # Rolling window size
```

---

## Step 7: Testing

1. **Compare Performance**: Train models with and without time series features
2. **Feature Importance**: Check which temporal features are most important
3. **Cross-Validation**: Ensure CV still works correctly (temporal order preserved)

---

## Expected Benefits

1. **Better Trend Detection**: Roll Wear change rate can predict TWF
2. **Anomaly Detection**: Sudden changes in speed/torque (acceleration features)
3. **Thermal Patterns**: Temperature trend can help predict HDF
4. **Power Patterns**: Rolling power statistics can help predict PWF

---

## Implementation Order

1. Add temporal feature engineering methods to `DataPreprocessor`
2. Update `engineer_features()` to call temporal methods
3. Handle NaN values from lag features
4. Test on training data
5. Update prediction function for live monitoring
6. Re-train models and compare performance
7. Update documentation

---

## Potential Issues & Solutions

### Issue 1: No Historical Data for Live Predictions
**Solution**: Use current values as lag features, or maintain prediction history in Streamlit session state

### Issue 2: Data Leakage in Cross-Validation
**Solution**: Ensure temporal order is preserved in CV folds (use TimeSeriesSplit if needed)

### Issue 3: High Correlation with Original Features
**Solution**: Existing correlation check will handle this automatically

### Issue 4: Increased Feature Count
**Solution**: Model selection will naturally choose important features, correlation check removes redundant ones

---

## Next Steps

Would you like me to:
1. **Implement the temporal features** in `preprocessor.py`?
2. **Test the implementation** and compare performance?
3. **Update the prediction function** for live monitoring?
