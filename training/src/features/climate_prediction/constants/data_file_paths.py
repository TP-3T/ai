"""Constants file for climate prediction data file paths."""

import os

__ABSOLUTE_PATH_TO_PROJECT_DIR: str = os.getcwd()

# --- FILES USED FOR INPUT

__INTERIM_DATA_DIR_NAME:  str = "interim"
__PROCESSED_DATA_DIR_NAME:  str = "processed"

CLIMATE_DATASET_FILENAME: str = os.path.join(
  "/", 
  __ABSOLUTE_PATH_TO_PROJECT_DIR, 
  "data", 
  "climate_prediction", 
  __INTERIM_DATA_DIR_NAME, 
  "Interim_Climate_Dataset_from_1940_baseline_1950_FINAL_absolute_no_change_values.csv"
)

PROCESSED_DATA_DIR_PATH: list[str] = [
  "/", 
  __ABSOLUTE_PATH_TO_PROJECT_DIR, 
  "data", 
  "climate_prediction", 
  __PROCESSED_DATA_DIR_NAME, 
]

TRAINING_CLIMATE_DATASET_FILEPATH: str = os.path.join(
  *PROCESSED_DATA_DIR_PATH,
  "climate_prediction_training.csv"
)

TESTING_CLIMATE_DATASET_FILEPATH: str = os.path.join(
  *PROCESSED_DATA_DIR_PATH,
  "climate_prediction_testing.csv"
)

#  --- FILES USED FOR OUTPUT ---

__MODEL_OUTPUT_DIR_PATH: list[str] = [
  "/", 
  __ABSOLUTE_PATH_TO_PROJECT_DIR, 
  "src",
  "features",
  "climate_prediction",
  "output"
]

__GBDT_MODEL_OUTPUT_DIR_PATH: list[str] = [
  *__MODEL_OUTPUT_DIR_PATH,
  "gbdt_model"
]

GBDT_MODEL_FILE_NAME: str = os.path.join(
  *__GBDT_MODEL_OUTPUT_DIR_PATH,
  "xgboost_model.onnx"
)

GBDT_FEATURE_MAPPING_FILE_NAME: str = os.path.join(
  *__GBDT_MODEL_OUTPUT_DIR_PATH,
  "xgboost_model_feat_mappings.json"
)



__POLYNOMIAL_MODEL_DIR_PATH: list[str] = [
  *__MODEL_OUTPUT_DIR_PATH,
  "polynomial_model"
]

POLYNOMIAL_TEMP_MODEL_FILENAME: str = os.path.join(
  *__POLYNOMIAL_MODEL_DIR_PATH,
  "polynomial_temperature_model.pkl"
)

POLYNOMIAL_SEA_LEVEL_MODEL_FILENAME: str = os.path.join(
  *__POLYNOMIAL_MODEL_DIR_PATH,
  "polynomial_sea_level_model.pkl"
)

POLYNOMIAL_CO2_MODEL_FILENAME: str = os.path.join(
  *__POLYNOMIAL_MODEL_DIR_PATH,
  "polynomial_co2_model.pkl"
)

POLYNOMIAL_VISUALIZATION_FILENAME: str = os.path.join(
  *__POLYNOMIAL_MODEL_DIR_PATH,
  "polynomial_regression_fits.png"
)