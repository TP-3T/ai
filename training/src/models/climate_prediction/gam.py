"""
Climate prediction model training using Generalized Additive Models (GAM).
For Turn The Tides Game - AI Team

This script trains GAM models to predict future climate conditions based on current state.
Predicts both temperature and sea level for the next year.
"""

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from pygam import LinearGAM, s # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import os



from constants.data_file_paths import CLIMATE_DATASET_FILENAME, GAM_TEMP_MODEL_FILENAME, GAM_SEA_LEVEL_MODEL_FILENAME, GAM_VISUALIZATION_FILENAME
from constants.climate_prediction_dataset_cols import CO2, FUTURE_SEA_LVL, FUTURE_TEMP, SEA_LVL, TEMP, YEAR

# Display constants
ACTUAL_TEMPERATURE = 'Actual Future Temperature (deg C)'
ACTUAL_SEA_LVL = 'Actual Future Sea level (mm)'
PREDICTED_TEMPERATURE = 'Predicted Future Temperature (deg C)'
PREDICTED_SEA_LVL = 'Predicted Sea level (mm)'

# Axis constants
COL_AXIS_NUM = 1

def load_climate_dataset(filepath: str) -> DataFrame:
    """
    Load the climate dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the climate data
    """
    climate_dataset = pd.read_csv(filepath)
    return climate_dataset


def train_validate_and_test_model(climate_dataset: DataFrame) -> tuple:
    """
    Train, validate, and test GAM models for climate prediction.
    
    Trains two separate models:
    1. Temperature prediction model
    2. Sea level prediction model
    
    Args:
        climate_dataset: DataFrame containing the climate data
        
    Returns:
        Tuple of (temperature_model, sea_level_model)
    """
    
    print("=" * 70)
    print("TURN THE TIDES - CLIMATE GAM TRAINING")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: DATA EXPLORATION
    # ========================================================================
    
    print("\n📊 Dataset Information:")
    print(f"Total rows: {len(climate_dataset)}")
    print(f"Columns: {list(climate_dataset.columns)}")
    print(f"Year range: {climate_dataset[YEAR].min()} to {climate_dataset[YEAR].max()}")
    
    print("\n📋 First few rows:")
    print(climate_dataset.head())
    
    print("\n📋 Last few rows:")
    print(climate_dataset.tail())
    
    print("\n📈 Basic Statistics:")
    print(climate_dataset.describe())
    
    # Check for missing values
    print("\n❓ Missing values:")
    print(climate_dataset.isnull().sum())
    
    # ========================================================================
    # STEP 2: PREPARE DATA
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 70)
    
    # Features: Current climate state (excluding Year and future values)
    features = climate_dataset.drop(columns=[YEAR, FUTURE_TEMP, FUTURE_SEA_LVL], axis=COL_AXIS_NUM)
    
    # Targets: Future temperature and sea level
    target_temp = climate_dataset[FUTURE_TEMP]
    target_sea_level = climate_dataset[FUTURE_SEA_LVL]
    
    # Split data into training and testing sets (75/25 split, same as GBDT)
    # Using random_state=1 for reproducibility
    feat_train, feat_test, temp_train, temp_test = train_test_split(
        features, target_temp, test_size=0.25, random_state=1
    )
    
    _, _, sea_train, sea_test = train_test_split(
        features, target_sea_level, test_size=0.25, random_state=1
    )
    
    print(f"\n✅ Training set size: {len(feat_train)} samples")
    print(f"✅ Testing set size: {len(feat_test)} samples")
    print(f"\nFeature columns used: {list(features.columns)}")
    
    # ========================================================================
    # STEP 3: TRAIN TEMPERATURE MODEL
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TRAINING GAM MODEL FOR TEMPERATURE PREDICTION")
    print("=" * 70)
    
    # Create GAM for temperature
    # s() creates smooth spline terms for each feature
    # Features: CO2, Current Temperature, Current Sea Level
    temp_gam = LinearGAM(
        s(0) +  # CO2 concentration
        s(1) +  # Current temperature
        s(2)    # Current sea level
    )
    
    print("\n🔄 Training temperature model...")
    temp_gam.gridsearch(feat_train.values, temp_train.values)
    
    print("✅ Temperature model training complete!")
    print(f"   Number of terms: {len(temp_gam.terms)}")
    print(f"   Lambda values: {temp_gam.lam}")
    
    # Make predictions
    temp_pred_train = temp_gam.predict(feat_train.values)
    temp_pred_test = temp_gam.predict(feat_test.values)
    
    # Evaluate temperature model
    train_r2_temp = r2_score(temp_train, temp_pred_train)
    test_r2_temp = r2_score(temp_test, temp_pred_test)
    train_rmse_temp = np.sqrt(mean_squared_error(temp_train, temp_pred_train))
    test_rmse_temp = np.sqrt(mean_squared_error(temp_test, temp_pred_test))
    test_mae_temp = mean_absolute_error(temp_test, temp_pred_test)
    
    print(f"\n📊 Temperature Model Performance:")
    print(f"   Training R²: {train_r2_temp:.5f}")
    print(f"   Testing R²: {test_r2_temp:.5f}")
    print(f"   Training RMSE: {train_rmse_temp:.5f} °C")
    print(f"   Testing RMSE: {test_rmse_temp:.5f} °C")
    print(f"   Testing MAE: {test_mae_temp:.5f} °C")
    
    # ========================================================================
    # STEP 4: TRAIN SEA LEVEL MODEL
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TRAINING GAM MODEL FOR SEA LEVEL PREDICTION")
    print("=" * 70)
    
    # Create GAM for sea level
    sea_level_gam = LinearGAM(
        s(0) +  # CO2 concentration
        s(1) +  # Current temperature
        s(2)    # Current sea level
    )
    
    print("\n🔄 Training sea level model...")
    sea_level_gam.gridsearch(feat_train.values, sea_train.values)
    
    print("✅ Sea level model training complete!")
    print(f"   Number of terms: {len(sea_level_gam.terms)}")
    print(f"   Lambda values: {sea_level_gam.lam}")
    
    # Make predictions
    sea_pred_train = sea_level_gam.predict(feat_train.values)
    sea_pred_test = sea_level_gam.predict(feat_test.values)
    
    # Evaluate sea level model
    train_r2_sea = r2_score(sea_train, sea_pred_train)
    test_r2_sea = r2_score(sea_test, sea_pred_test)
    train_rmse_sea = np.sqrt(mean_squared_error(sea_train, sea_pred_train))
    test_rmse_sea = np.sqrt(mean_squared_error(sea_test, sea_pred_test))
    test_mae_sea = mean_absolute_error(sea_test, sea_pred_test)
    
    print(f"\n📊 Sea Level Model Performance:")
    print(f"   Training R²: {train_r2_sea:.5f}")
    print(f"   Testing R²: {test_r2_sea:.5f}")
    print(f"   Training RMSE: {train_rmse_sea:.5f} mm")
    print(f"   Testing RMSE: {test_rmse_sea:.5f} mm")
    print(f"   Testing MAE: {test_mae_sea:.5f} mm")
    
    # ========================================================================
    # STEP 5: DISPLAY TEST RESULTS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Testing Dataset Prediction Results:")
    print("=" * 70)
    
    # Create results dataframe (similar to GBDT output)
    test_results = pd.DataFrame(index=feat_test.index)
    test_results[ACTUAL_TEMPERATURE] = temp_test.values
    test_results[PREDICTED_TEMPERATURE] = temp_pred_test
    test_results[ACTUAL_SEA_LVL] = sea_test.values
    test_results[PREDICTED_SEA_LVL] = sea_pred_test
    
    print(f"\n{test_results}")
    
    print(f"\nRMSE (average error) of temperature model: {test_rmse_temp:.5f} °C")
    print(f"RMSE (average error) of sea level model: {test_rmse_sea:.5f} mm")
    print(f"Model goodness of fit (Temperature R²): {test_r2_temp:.5f}")
    print(f"Model goodness of fit (Sea Level R²): {test_r2_sea:.5f}")
    
    # ========================================================================
    # STEP 6: VISUALIZE RESULTS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature: Actual vs Predicted (Training)
    axes[0, 0].scatter(temp_train, temp_pred_train, alpha=0.5, s=20)
    axes[0, 0].plot([temp_train.min(), temp_train.max()], 
                    [temp_train.min(), temp_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Temperature (°C)')
    axes[0, 0].set_ylabel('Predicted Temperature (°C)')
    axes[0, 0].set_title(f'Temperature Model - Training\nR² = {train_r2_temp:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperature: Actual vs Predicted (Testing)
    axes[0, 1].scatter(temp_test, temp_pred_test, alpha=0.5, s=20, color='green')
    axes[0, 1].plot([temp_test.min(), temp_test.max()], 
                    [temp_test.min(), temp_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Temperature (°C)')
    axes[0, 1].set_ylabel('Predicted Temperature (°C)')
    axes[0, 1].set_title(f'Temperature Model - Testing\nR² = {test_r2_temp:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sea Level: Actual vs Predicted (Training)
    axes[1, 0].scatter(sea_train, sea_pred_train, alpha=0.5, s=20)
    axes[1, 0].plot([sea_train.min(), sea_train.max()], 
                    [sea_train.min(), sea_train.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Sea Level (mm)')
    axes[1, 0].set_ylabel('Predicted Sea Level (mm)')
    axes[1, 0].set_title(f'Sea Level Model - Training\nR² = {train_r2_sea:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sea Level: Actual vs Predicted (Testing)
    axes[1, 1].scatter(sea_test, sea_pred_test, alpha=0.5, s=20, color='green')
    axes[1, 1].plot([sea_test.min(), sea_test.max()], 
                    [sea_test.min(), sea_test.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Sea Level (mm)')
    axes[1, 1].set_ylabel('Predicted Sea Level (mm)')
    axes[1, 1].set_title(f'Sea Level Model - Testing\nR² = {test_r2_sea:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GAM_VISUALIZATION_FILENAME, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to '{GAM_VISUALIZATION_FILENAME}'")
    
    return temp_gam, sea_level_gam


def visualize_dataset(climate_dataset: DataFrame):
    """
    Visualize the climate dataset relationships.
    Creates multiple plots showing how variables relate to each other.
    """
    
    year_col = climate_dataset[YEAR]
    temp_col = climate_dataset[TEMP]
    sea_lvl_col = climate_dataset[SEA_LVL]
    co2_col = climate_dataset[CO2]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    
    # Row 0: Time series plots
    axs[0, 0].plot(year_col, temp_col, color='blue')
    axs[0, 0].set_title(f"{YEAR} vs Temperature")
    axs[0, 0].set_xlabel(YEAR)
    axs[0, 0].set_ylabel('Temperature (°C)')
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(year_col, sea_lvl_col, color='red')
    axs[0, 1].set_title(f"{YEAR} vs Sea Level")
    axs[0, 1].set_xlabel(YEAR)
    axs[0, 1].set_ylabel('Sea Level (mm)')
    axs[0, 1].grid(True, alpha=0.3)
    
    axs[0, 2].plot(year_col, co2_col, color='green')
    axs[0, 2].set_title(f"{YEAR} vs CO2")
    axs[0, 2].set_xlabel(YEAR)
    axs[0, 2].set_ylabel('CO2 (ppm)')
    axs[0, 2].grid(True, alpha=0.3)
    
    # Row 1: CO2 relationships
    axs[1, 0].scatter(co2_col, temp_col, alpha=0.5, color='blue')
    axs[1, 0].set_title("CO2 vs Temperature")
    axs[1, 0].set_xlabel('CO2 (ppm)')
    axs[1, 0].set_ylabel('Temperature (°C)')
    axs[1, 0].grid(True, alpha=0.3)
    
    axs[1, 1].scatter(co2_col, sea_lvl_col, alpha=0.5, color='red')
    axs[1, 1].set_title("CO2 vs Sea Level")
    axs[1, 1].set_xlabel('CO2 (ppm)')
    axs[1, 1].set_ylabel('Sea Level (mm)')
    axs[1, 1].grid(True, alpha=0.3)
    
    axs[1, 2].scatter(temp_col, sea_lvl_col, alpha=0.5, color='green')
    axs[1, 2].set_title("Temperature vs Sea Level")
    axs[1, 2].set_xlabel('Temperature (°C)')
    axs[1, 2].set_ylabel('Sea Level (mm)')
    axs[1, 2].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(3):
        axs[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Loading climate dataset...")
    climate_dataset = load_climate_dataset(CLIMATE_DATASET_FILENAME)
    
    # Train and test the models
    temp_model, sea_level_model = train_validate_and_test_model(climate_dataset)
    
    # Save models
    print("\n" + "=" * 70)
    print("SAVING TRAINED MODELS")
    print("=" * 70)
    
    with open(GAM_TEMP_MODEL_FILENAME, 'wb') as f:
        pickle.dump(temp_model, f)
    print(f"✅ Temperature model saved to '{GAM_TEMP_MODEL_FILENAME}'")
    
    with open(GAM_SEA_LEVEL_MODEL_FILENAME, 'wb') as f:
        pickle.dump(sea_level_model, f)
    print(f"✅ Sea level model saved to '{GAM_SEA_LEVEL_MODEL_FILENAME}'")
    
    # Optionally visualize the dataset
    # Uncomment the line below to see data visualizations
    # visualize_dataset(climate_dataset)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel files created:")
    print(f"  - {GAM_TEMP_MODEL_FILENAME}")
    print(f"  - {GAM_SEA_LEVEL_MODEL_FILENAME}")
    print(f"  - {GAM_VISUALIZATION_FILENAME}")