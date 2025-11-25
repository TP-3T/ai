"""
Climate prediction model training using polynomial regression.
Contains functions to visualize the climate data, as well as perform the model training, validation and testing.
This model is intended for interpolation within the historical data range (1950-2024).
For extrapolation beyond this range, switch to the physics-based model.
"""

# Polynomial Regression:
#   - A form of regression analysis where the relationship between features and target is modeled as an nth degree polynomial
#   - Unlike linear regression (which fits a straight line), polynomial regression can capture non-linear relationships
#   - The model fits a curve to the data points
#   - Example: y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
#   - Where n is the degree of the polynomial

# Polynomial Regression for Climate Data:
#   - Climate relationships are often non-linear (e.g., CO2 vs Temperature)
#   - Polynomial regression can capture these curved relationships
#   - However, polynomials are POOR for extrapolation beyond the training data range
#   - Best used for interpolation (predicting within the known data range)

# Model Selection Criteria:
#   - Degree selection is critical:
#       - Too low (underfitting): Model too simple to capture patterns
#       - Too high (overfitting): Model memorizes noise, poor generalization
#   - Common degrees: 2 (quadratic), 3 (cubic), 4 (quartic)
#   - Use cross-validation and metrics (R², RMSE) to select optimal degree

# Train/Validation/Test Split:
#   - Training set: Used to fit the polynomial coefficients
#   - Validation set: Used to select the best polynomial degree
#   - Test set: Final evaluation on completely unseen data

# =============================================================================

import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple

# Import file paths from constants file
from constants.data_file_paths import (  # type: ignore
    INTERIM_CLIMATE_DATASET_FROM_1950_FILENAME,
    POLYNOMIAL_TEMP_MODEL_FILENAME,
    POLYNOMIAL_SEA_LEVEL_MODEL_FILENAME,
    # POLYNOMIAL_CO2_MODEL_FILENAME,  # Not used - CO2 prediction removed
    POLYNOMIAL_VISUALIZATION_FILENAME
)

# Assign to shorter variable names for convenience
CLIMATE_DATASET_FILENAME = INTERIM_CLIMATE_DATASET_FROM_1950_FILENAME
POLY_TEMPERATURE_MODEL_FILEPATH = POLYNOMIAL_TEMP_MODEL_FILENAME
POLY_SEA_LEVEL_MODEL_FILEPATH = POLYNOMIAL_SEA_LEVEL_MODEL_FILENAME
# POLY_CO2_MODEL_FILEPATH = POLYNOMIAL_CO2_MODEL_FILENAME  # Not used
POLY_VISUALIZATION_FILEPATH = POLYNOMIAL_VISUALIZATION_FILENAME

# Column names from the dataset
YEAR = "Year"
SEA_LVL = "absolute gmsl (mm) relative to 1940"
CO2 = "CO2 average (ppm)"
TEMP = "Temperature (deg C)"

# Target variable column names (must match dataset exactly!)
FUTURE_TEMP = "Temperature change 3 months ago"
FUTURE_SEA_LVL = "gmsl change in 3 months "  # Note: has trailing space in dataset!
FUTURE_CO2 = "CO2 change 3 months ago"

# Display names for results
ACTUAL_TEMPERATURE = "Actual Temperature Change"
PREDICTED_TEMPERATURE = "Predicted Temperature Change"
ACTUAL_SEA_LVL = "Actual Sea Level Change"
PREDICTED_SEA_LVL = "Predicted Sea Level Change"
ACTUAL_CO2 = "Actual CO2 Change"
PREDICTED_CO2 = "Predicted CO2 Change"

# Configuration constants
TEST_SIZE = 0.1       # 10% for final testing
VALIDATION_SIZE = 0.2  # 20% for validation (from remaining 90%)
RANDOM_STATE = 42
COL_AXIS_NUM = 1

# Polynomial degree settings - adjust based on cross-validation results
TEMPERATURE_POLY_DEGREE = 3  # Cubic polynomial for temperature
SEA_LEVEL_POLY_DEGREE = 2    # Quadratic polynomial for sea level
CO2_POLY_DEGREE = 2          # Quadratic polynomial for CO2

# Training data range thresholds (for documentation purposes)
CO2_MIN_TRAINING = 280.0    # Minimum CO2 in training data (ppm)
CO2_MAX_TRAINING = 420.0    # Maximum CO2 in training data (ppm)
TEMP_MIN_TRAINING = 13.8    # Minimum temperature in training data (°C)
TEMP_MAX_TRAINING = 14.7    # Maximum temperature in training data (°C)

# Messages
LOADING_DATASET_MSG = "Loading climate dataset"
DATASET_LOADED_MSG = "Climate dataset loaded successfully"
TRAINING_MODEL_MSG = "Training polynomial regression model"
MODEL_TRAINED_MSG = "Model training complete"
TEST_PREDICTION_RESULTS_MSG = "Test Set Predictions vs Actual Values"
RMSE_TEMP_REPORT_MSG = "Temperature Model RMSE"
RMSE_SEA_LVL_REPORT_MSG = "Sea Level Model RMSE"
RMSE_CO2_REPORT_MSG = "CO2 Model RMSE"
COEFF_OF_DET_TEMP_REPORT_MSG = "Temperature Model R² Score"
COEFF_OF_DET_SEA_LVL_REPORT_MSG = "Sea Level Model R² Score"
COEFF_OF_DET_CO2_REPORT_MSG = "CO2 Model R² Score"
SAVING_MODEL_MSG = "Saving trained model to"
MODEL_SAVED_MSG = "Model saved successfully"

# =============================================================================


def __load_climate_dataset() -> DataFrame:
    """
    Load the climate dataset from CSV file.
    
    Returns:
        DataFrame: The loaded climate dataset
    """
    print(f"\n{LOADING_DATASET_MSG}...")
    
    # Load dataset - adjust path as needed for your project structure
    climate_dataset = pd.read_csv(CLIMATE_DATASET_FILENAME, encoding='utf-8-sig')
    
    print(f"{DATASET_LOADED_MSG}")
    print(f"Dataset shape: {climate_dataset.shape}")
    print(f"Columns: {list(climate_dataset.columns)}\n")
    
    return climate_dataset


def __split_dataset_into_train_val_test(
    features: DataFrame,
    target: DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """
    Split the dataset into training (70%), validation (20%), and testing (10%) sets.
    
    Args:
        features: Feature columns (X)
        target: Target column (y)
        test_size: Proportion of data to use for testing (0.1 = 10%)
        val_size: Proportion of remaining data to use for validation (0.2 of 90% = 18% total)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate out test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: split remaining into train (70%) and validation (20%)
    # val_size / (1 - test_size) gives us the right proportion
    # e.g., 0.2 / 0.9 ≈ 0.222, which takes 20% of the remaining 90%
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def __train_polynomial_model(
    X_train: DataFrame,
    y_train: Series,
    poly_degree: int
) -> Tuple[PolynomialFeatures, LinearRegression]:
    """
    Train a polynomial regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        poly_degree: Degree of the polynomial
    
    Returns:
        Tuple of (polynomial_features_transformer, trained_model)
    """
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_train_poly = poly_features.fit_transform(X_train)
    
    # Train linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    return poly_features, model


def __predict_with_polynomial_model(
    model: LinearRegression,
    poly_features: PolynomialFeatures,
    X: DataFrame
) -> np.ndarray:
    """
    Make predictions using a trained polynomial regression model.
    
    Args:
        model: Trained linear regression model
        poly_features: Polynomial features transformer
        X: Features to predict on
    
    Returns:
        Array of predictions
    """
    X_poly = poly_features.transform(X)
    predictions = model.predict(X_poly)
    
    return predictions


def __save_model(
    poly_features: PolynomialFeatures,
    model: LinearRegression,
    filepath: str
) -> None:
    """
    Save the polynomial regression model and its transformer to a pickle file.
    
    Args:
        poly_features: Polynomial features transformer
        model: Trained model
        filepath: Path to save the model
    """
    print(f"{SAVING_MODEL_MSG}: {filepath}")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_package = {
        'poly_features': poly_features,
        'model': model,
        'poly_degree': poly_features.degree
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"{MODEL_SAVED_MSG}\n")


def train_validate_and_test_model(climate_dataset: DataFrame) -> Tuple[dict, dict]:
    """
    Train polynomial regression models for temperature and sea level predictions.
    Uses 70% training, 20% validation, 10% test split.
    Evaluate models on validation and test sets and return the trained models.
    
    Args:
        climate_dataset: The complete climate dataset
    
    Returns:
        Tuple of (temperature_model_dict, sea_level_model_dict)
        Each dict contains 'poly_features' and 'model' keys
    """
    print(f"\n{'='*80}")
    print(f"{TRAINING_MODEL_MSG}")
    print(f"{'='*80}\n")
    
    # Prepare features - using CO2 as primary feature
    # You can add more features here if needed (e.g., Year, previous temperature, etc.)
    features = climate_dataset[[CO2]].copy()
    
    # IMPORTANT: Predicting ABSOLUTE VALUES instead of CHANGES
    # Changes are too small and noisy for polynomial regression to predict well
    # If you need changes, calculate them from the predictions
    temp_target = climate_dataset[TEMP].copy()          # Absolute temperature
    sea_lvl_target = climate_dataset[SEA_LVL].copy()    # Absolute sea level
    
    print("NOTE: Training models to predict ABSOLUTE values, not changes.")
    print("This provides much better R² scores and more stable predictions.")
    print("Calculate changes from predictions if needed for your game.\n")
    
    # Split data for each target (70% train, 20% validation, 10% test)
    X_train_temp, X_val_temp, X_test_temp, y_train_temp, y_val_temp, y_test_temp = \
        __split_dataset_into_train_val_test(features, temp_target)
    
    X_train_sea, X_val_sea, X_test_sea, y_train_sea, y_val_sea, y_test_sea = \
        __split_dataset_into_train_val_test(features, sea_lvl_target)
    
    # Show dataset split sizes
    total_samples = len(climate_dataset)
    train_samples = len(X_train_temp)
    val_samples = len(X_val_temp)
    test_samples = len(X_test_temp)
    
    print(f"Dataset Split:")
    print(f"  Total samples: {total_samples}")
    print(f"  Training:   {train_samples} samples ({train_samples/total_samples*100:.1f}%)")
    print(f"  Validation: {val_samples} samples ({val_samples/total_samples*100:.1f}%)")
    print(f"  Test:       {test_samples} samples ({test_samples/total_samples*100:.1f}%)\n")
    
    # Store actual test data range for validation
    train_co2_min = features.min().values[0]
    train_co2_max = features.max().values[0]
    
    print(f"Training Data Range:")
    print(f"  CO2: {train_co2_min:.2f} - {train_co2_max:.2f} ppm")
    print(f"  Temperature: {climate_dataset[TEMP].min():.2f} - {climate_dataset[TEMP].max():.2f} °C")
    print(f"  Sea Level: {climate_dataset[SEA_LVL].min():.2f} - {climate_dataset[SEA_LVL].max():.2f} mm")
    print(f"\nWARNING: These models should ONLY be used for predictions within this range!")
    print(f"For predictions outside this range, use the physics-based model.\n")
    
    # Train Temperature Model
    print(f"Training Temperature Model (Polynomial Degree: {TEMPERATURE_POLY_DEGREE})...")
    poly_features_temp, temp_model = __train_polynomial_model(
        X_train_temp, y_train_temp, TEMPERATURE_POLY_DEGREE
    )
    
    # Train Sea Level Model
    print(f"Training Sea Level Model (Polynomial Degree: {SEA_LEVEL_POLY_DEGREE})...")
    poly_features_sea, sea_lvl_model = __train_polynomial_model(
        X_train_sea, y_train_sea, SEA_LEVEL_POLY_DEGREE
    )
    
    print(f"\n{MODEL_TRAINED_MSG}\n")
    
    # Make predictions on VALIDATION set
    predicted_temps_val = __predict_with_polynomial_model(temp_model, poly_features_temp, X_val_temp)
    predicted_sea_lvls_val = __predict_with_polynomial_model(sea_lvl_model, poly_features_sea, X_val_sea)
    
    # Make predictions on TEST set
    predicted_temps_test = __predict_with_polynomial_model(temp_model, poly_features_temp, X_test_temp)
    predicted_sea_lvls_test = __predict_with_polynomial_model(sea_lvl_model, poly_features_sea, X_test_sea)
    
    # Create results DataFrames for TEST set
    temp_results = pd.DataFrame({
        'Actual Temperature': y_test_temp.values,
        'Predicted Temperature': predicted_temps_test
    }, index=y_test_temp.index)
    
    sea_lvl_results = pd.DataFrame({
        'Actual Sea Level': y_test_sea.values,
        'Predicted Sea Level': predicted_sea_lvls_test
    }, index=y_test_sea.index)
    
    # Display test results
    print(f"\n{TEST_PREDICTION_RESULTS_MSG}:\n")
    print("Temperature Predictions (Test Set):")
    print(temp_results.head(10))
    print(f"\nSea Level Predictions (Test Set):")
    print(sea_lvl_results.head(10))
    
    # Model Evaluation - Calculate metrics for BOTH validation and test sets
    print(f"\n{'='*80}")
    print("MODEL EVALUATION METRICS")
    print(f"{'='*80}\n")
    
    # Validation set metrics
    temp_rmse_val = root_mean_squared_error(y_val_temp.values, predicted_temps_val)
    sea_lvl_rmse_val = root_mean_squared_error(y_val_sea.values, predicted_sea_lvls_val)
    
    temp_r2_val = r2_score(y_val_temp.values, predicted_temps_val)
    sea_lvl_r2_val = r2_score(y_val_sea.values, predicted_sea_lvls_val)
    
    # Test set metrics
    temp_rmse_test = root_mean_squared_error(y_test_temp.values, predicted_temps_test)
    sea_lvl_rmse_test = root_mean_squared_error(y_test_sea.values, predicted_sea_lvls_test)
    
    temp_r2_test = r2_score(y_test_temp.values, predicted_temps_test)
    sea_lvl_r2_test = r2_score(y_test_sea.values, predicted_sea_lvls_test)
    
    print("VALIDATION SET Performance:")
    print(f"  {RMSE_TEMP_REPORT_MSG}: {temp_rmse_val:.5f}")
    print(f"  {RMSE_SEA_LVL_REPORT_MSG}: {sea_lvl_rmse_val:.5f}\n")
    
    print(f"  {COEFF_OF_DET_TEMP_REPORT_MSG}: {temp_r2_val:.5f} ({temp_r2_val*100:.2f}%)")
    print(f"  {COEFF_OF_DET_SEA_LVL_REPORT_MSG}: {sea_lvl_r2_val:.5f} ({sea_lvl_r2_val*100:.2f}%)\n")
    
    print("TEST SET Performance (Final Evaluation):")
    print(f"  {RMSE_TEMP_REPORT_MSG}: {temp_rmse_test:.5f}")
    print(f"  {RMSE_SEA_LVL_REPORT_MSG}: {sea_lvl_rmse_test:.5f}\n")
    
    print(f"  {COEFF_OF_DET_TEMP_REPORT_MSG}: {temp_r2_test:.5f} ({temp_r2_test*100:.2f}%)")
    print(f"  {COEFF_OF_DET_SEA_LVL_REPORT_MSG}: {sea_lvl_r2_test:.5f} ({sea_lvl_r2_test*100:.2f}%)\n")
    
    # Interpretation
    print("Interpretation:")
    print(f"  - R² close to 1.0 indicates excellent fit")
    print(f"  - R² > 0.9 indicates very good fit")
    print(f"  - R² between 0.7-0.9 indicates good fit")
    print(f"  - R² < 0.7 may indicate need for higher degree polynomial or different features")
    print(f"  - Validation and test scores should be similar (no overfitting)\n")
    
    # Save models
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}\n")
    
    __save_model(poly_features_temp, temp_model, POLY_TEMPERATURE_MODEL_FILEPATH)
    __save_model(poly_features_sea, sea_lvl_model, POLY_SEA_LEVEL_MODEL_FILEPATH)
    
    # Package models for return
    temp_model_package = {
        'poly_features': poly_features_temp,
        'model': temp_model,
        'poly_degree': TEMPERATURE_POLY_DEGREE
    }
    
    sea_lvl_model_package = {
        'poly_features': poly_features_sea,
        'model': sea_lvl_model,
        'poly_degree': SEA_LEVEL_POLY_DEGREE
    }
    
    return temp_model_package, sea_lvl_model_package


def visualize_polynomial_fit(climate_dataset: DataFrame):
    """
    Visualize the polynomial regression fits for temperature, sea level, and CO2.
    Shows the actual data points and the fitted polynomial curves.
    """
    # Prepare data - using absolute values
    co2_data = climate_dataset[CO2].values
    temp_data = climate_dataset[TEMP].values
    sea_lvl_data = climate_dataset[SEA_LVL].values
    
    # Create polynomial features for visualization
    co2_range = np.linspace(co2_data.min(), co2_data.max(), 300).reshape(-1, 1)
    
    # Fit models for visualization
    poly_temp = PolynomialFeatures(degree=TEMPERATURE_POLY_DEGREE)
    model_temp = LinearRegression()
    X_poly_temp = poly_temp.fit_transform(co2_data.reshape(-1, 1))
    model_temp.fit(X_poly_temp, temp_data)
    temp_pred_curve = model_temp.predict(poly_temp.transform(co2_range))
    
    poly_sea = PolynomialFeatures(degree=SEA_LEVEL_POLY_DEGREE)
    model_sea = LinearRegression()
    X_poly_sea = poly_sea.fit_transform(co2_data.reshape(-1, 1))
    model_sea.fit(X_poly_sea, sea_lvl_data)
    sea_pred_curve = model_sea.predict(poly_sea.transform(co2_range))
    
    # Create visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temperature plot
    axs[0].scatter(co2_data, temp_data, alpha=0.5, s=10, label='Actual Data')
    axs[0].plot(co2_range, temp_pred_curve, 'r-', linewidth=2, 
                label=f'Polynomial Fit (degree={TEMPERATURE_POLY_DEGREE})')
    axs[0].set_xlabel('CO2 (ppm)')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Temperature vs CO2')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Sea level plot
    axs[1].scatter(co2_data, sea_lvl_data, alpha=0.5, s=10, label='Actual Data')
    axs[1].plot(co2_range, sea_pred_curve, 'b-', linewidth=2, 
                label=f'Polynomial Fit (degree={SEA_LEVEL_POLY_DEGREE})')
    axs[1].set_xlabel('CO2 (ppm)')
    axs[1].set_ylabel('Sea Level (mm)')
    axs[1].set_title('Sea Level vs CO2')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(POLY_VISUALIZATION_FILEPATH, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{POLY_VISUALIZATION_FILEPATH}'")
    plt.show()


def run_polynomial_regression() -> Tuple[dict, dict, dict, DataFrame]:
    """
    Main function to load data, train models, and return trained model packages.
    
    Returns:
        Tuple of (temp_model, sea_level_model, dataset)
    """
    print(f"\n{'='*80}")
    print("POLYNOMIAL REGRESSION CLIMATE PREDICTION MODEL")
    print(f"{'='*80}\n")
    
    climate_dataset = __load_climate_dataset()
    
    temp_model, sea_lvl_model = train_validate_and_test_model(climate_dataset)
    
    # Optional: Visualize the fits
    # visualize_polynomial_fit(climate_dataset)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Models saved to:")
    print(f"  - {POLY_TEMPERATURE_MODEL_FILEPATH}")
    print(f"  - {POLY_SEA_LEVEL_MODEL_FILEPATH}")
    
    return temp_model, sea_lvl_model, climate_dataset


if __name__ == "__main__":
    # Run the polynomial regression training
    temp_model, sea_lvl_model, dataset = run_polynomial_regression()
    
    # Optional: Uncomment to visualize the polynomial fits
    # visualize_polynomial_fit(dataset)