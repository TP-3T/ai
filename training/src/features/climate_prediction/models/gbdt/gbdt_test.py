from features.climate_prediction.data_processing.separate_features_and_targets import separate_features_and_targets_from_climate_dataset # type: ignore
from sklearn.metrics import r2_score, root_mean_squared_error # type: ignore
from features.climate_prediction.constants.data_file_paths import TESTING_CLIMATE_DATASET_FILEPATH # type: ignore
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

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
  for actual, predicted in zip(climate_dataset_targets_test.values[:10], predicted_targets_test[:10]):
    print(f"{float(actual):.4f}\t{float(predicted):.4f}")

  rmse: float = root_mean_squared_error(climate_dataset_targets_test, predicted_targets_test)
  r2: float = r2_score(climate_dataset_targets_test, predicted_targets_test)

  print("\n=== Test set performance ===")
  print(f"RMSE: {rmse:.4f}")
  print(f"R²  : {r2:.4f}")
