from features.climate_prediction.data_processing.separate_features_and_targets import separate_features_and_targets_from_climate_dataset # type: ignore
from sklearn.metrics import r2_score, root_mean_squared_error # type: ignore
from features.climate_prediction.constants.data_file_paths import TESTING_CLIMATE_DATASET_FILEPATH # type: ignore
from features.climate_prediction.constants.data_file_paths import TRAINING_CLIMATE_DATASET_FILEPATH # type: ignore
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
import numpy as np

# === Evaluate once on the test set === 

def test_gbdt_model(gbdt_model: Pipeline, target_col_name: str):
  """
  Test the gbdt model on the testing set using RMSE.
  """
  # === Load the testing climate prediction dataset ===
  testing_climate_dataset: DataFrame = pd.read_csv(TESTING_CLIMATE_DATASET_FILEPATH) # type: ignore

  # === Seperate features and targets from training dataset ===
  (climate_dataset_features_test, \
   climate_dataset_targets_test) = separate_features_and_targets_from_climate_dataset(testing_climate_dataset, target_col_name)

  # === Predict features from test set ===
  predicted_targets_test = gbdt_model.predict(climate_dataset_features_test) # type: ignore

  print("\n=== Sample predictions vs actuals (test set) ===")
  print("Actual\tPredicted")
  for actual, predicted in zip(climate_dataset_targets_test.values, predicted_targets_test):
    print(f"{float(actual):.4f}\t{float(predicted):.4f}")

  rmse: float = root_mean_squared_error(climate_dataset_targets_test, predicted_targets_test)
  r2: float = r2_score(climate_dataset_targets_test, predicted_targets_test)

  print("\n=== Test set performance ===")
  print(f"RMSE: {rmse:.4f}")
  print(f"R²  : {r2:.4f}")


# === Below is for other testing ===


def test_gbdt_model_interpolation(gbdt_model: Pipeline, target_col_name: str, num_samples: int = 100):
  """
  Test the model on synthetically generated data within its training range (interpolation).
  Generates random values within the min/max range of each feature from the training dataset,
  then uses the model to predict the target values.
  """
  
  training_climate_dataset: DataFrame = pd.read_csv(TRAINING_CLIMATE_DATASET_FILEPATH)
  
  (training_features, training_targets) = separate_features_and_targets_from_climate_dataset(
    training_climate_dataset, target_col_name
  )
  
  feature_names = training_features.columns.tolist()
  
  synthetic_features = pd.DataFrame()
  
  print("\n=== Generating synthetic interpolation test data ===")
  print(f"Creating {num_samples} synthetic samples within training range...")
  
  for feature_name in feature_names:
    min_val = training_features[feature_name].min()
    max_val = training_features[feature_name].max()
    
    synthetic_features[feature_name] = np.random.uniform(
      low=min_val,
      high=max_val,
      size=num_samples
    )
  
  predicted_targets_interpolation = gbdt_model.predict(synthetic_features)
  
  print("\n=== Synthetic feature samples (first 5) ===")
  print(synthetic_features.head())
  
  print("\n=== Model predictions on synthetic data (first 20) ===")
  print("Index\tPredicted Value")
  for i, predicted in enumerate(predicted_targets_interpolation[:20]):
    print(f"{i}\t{float(predicted):.4f}")
  
  pred_min = predicted_targets_interpolation.min()
  pred_max = predicted_targets_interpolation.max()
  pred_mean = predicted_targets_interpolation.mean()
  pred_std = predicted_targets_interpolation.std()
  
  training_target_min = training_targets.min()
  training_target_max = training_targets.max()
  training_target_mean = training_targets.mean()
  
  if isinstance(training_target_min, pd.Series):
    training_target_min = training_target_min.values[0]
    training_target_max = training_target_max.values[0]
    training_target_mean = training_target_mean.values[0]
  
  print(f"\n=== Interpolation prediction statistics ===")
  print(f"Number of synthetic samples: {num_samples}")
  print(f"\nPredicted target range:")
  print(f"  Min:  {pred_min:.4f}")
  print(f"  Max:  {pred_max:.4f}")
  print(f"  Mean: {pred_mean:.4f}")
  print(f"  Std:  {pred_std:.4f}")
  
  print(f"\nTraining target range (for comparison):")
  print(f"  Min:  {training_target_min:.4f}")
  print(f"  Max:  {training_target_max:.4f}")
  print(f"  Mean: {training_target_mean:.4f}")
  
  within_range_count = np.sum(
    (predicted_targets_interpolation >= training_target_min) & 
    (predicted_targets_interpolation <= training_target_max)
  )
  within_range_pct = (within_range_count / num_samples) * 100
  
  print(f"\nPredictions within training range: {within_range_count}/{num_samples} ({within_range_pct:.1f}%)")
  
  return synthetic_features, predicted_targets_interpolation



def test_multiple_synthetic_points(gbdt_model: Pipeline, target_col_name: str, num_points: int = 100):
  """
  Generate multiple synthetic data points and show their predictions in a table format.
  Includes R² and RMSE statistics by comparing predictions to similar points in training data.
  """
  
  training_climate_dataset: DataFrame = pd.read_csv(TRAINING_CLIMATE_DATASET_FILEPATH)
  
  (training_features, training_targets) = separate_features_and_targets_from_climate_dataset(
    training_climate_dataset, target_col_name
  )
  
  feature_names = training_features.columns.tolist()
  
  synthetic_points = []
  
  print(f"\n=== Generating {num_points} synthetic data points ===\n")
  
  for i in range(num_points):
    synthetic_point = {}
    
    for feature_name in feature_names:
      min_val = training_features[feature_name].min()
      max_val = training_features[feature_name].max()
      
      synthetic_value = np.random.uniform(low=min_val, high=max_val)
      synthetic_point[feature_name] = synthetic_value
    
    synthetic_points.append(synthetic_point)
  
  synthetic_df = pd.DataFrame(synthetic_points)
  
  predictions = gbdt_model.predict(synthetic_df)
  
  display_count = min(10, num_points)
  print("="*100)
  print(f"{'Point':<8} {'Key Features':<60} {'Prediction':<15}")
  print("="*100)
  
  key_features = ['CO2 (ppm)', 'TEMP (deg C)', 'Absolute GMSL (mm) relative to Jan 1950']
  key_features = [f for f in key_features if f in synthetic_df.columns]
  
  for i in range(display_count):
    feature_summary = " | ".join([f"{synthetic_df.iloc[i][feat]:.2f}" for feat in key_features])
    print(f"{i+1:<8} {feature_summary:<60} {predictions[i]:.4f} mm")
  
  if num_points > display_count:
    print(f"... ({num_points - display_count} more points)")
  
  print("="*100)
  print(f"\nColumn order: {' | '.join(key_features)}")
  
  from sklearn.metrics import r2_score, root_mean_squared_error
  
  actual_predictions_on_training = gbdt_model.predict(training_features)
  
  training_target_min = training_targets.min()
  training_target_max = training_targets.max()
  training_target_mean = training_targets.mean()
  
  if isinstance(training_target_min, pd.Series):
    training_target_min = training_target_min.values[0]
    training_target_max = training_target_max.values[0]
    training_target_mean = training_target_mean.values[0]
  
  if isinstance(training_targets, pd.DataFrame):
    training_targets_values = training_targets.values.ravel()
  else:
    training_targets_values = training_targets.values
  
  r2_training = r2_score(training_targets_values, actual_predictions_on_training)
  rmse_training = root_mean_squared_error(training_targets_values, actual_predictions_on_training)
  
  pred_min = predictions.min()
  pred_max = predictions.max()
  pred_mean = predictions.mean()
  pred_std = predictions.std()
  
  within_range_count = np.sum(
    (predictions >= training_target_min) & 
    (predictions <= training_target_max)
  )
  within_range_pct = (within_range_count / num_points) * 100
  
  print("\n" + "="*100)
  print("PREDICTION STATISTICS")
  print("="*100)
  
  print(f"\nSynthetic predictions:")
  print(f"  Min:  {pred_min:.4f} mm")
  print(f"  Max:  {pred_max:.4f} mm")
  print(f"  Mean: {pred_mean:.4f} mm")
  print(f"  Std:  {pred_std:.4f} mm")
  
  print(f"\nTraining data range (for reference):")
  print(f"  Min:  {training_target_min:.4f} mm")
  print(f"  Max:  {training_target_max:.4f} mm")
  print(f"  Mean: {training_target_mean:.4f} mm")
  
  print(f"\nModel performance on training data (interpolation baseline):")
  print(f"  R²:   {r2_training:.4f} ({r2_training*100:.2f}%)")
  print(f"  RMSE: {rmse_training:.4f} mm")
  
  print(f"\nSynthetic data validation:")
  print(f"  Predictions within training range: {within_range_count}/{num_points} ({within_range_pct:.1f}%)")
  
  if within_range_count == num_points:
    print(f"  ✓ All synthetic predictions are realistic (within training range)")
  elif within_range_pct >= 90:
    print(f"  ✓ Most synthetic predictions are realistic")
  else:
    print(f"  ⚠ Warning: Some predictions fall outside training range")
  
  print("="*100)
  
  return synthetic_df, predictions