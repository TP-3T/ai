"""Constants file for data file paths."""

import os

# === CLIMATE PREDICTION ===
# __TEMP_FILE_NAME:     str = "Indicator_3_1_Climate_Annual_Mean_Global_Surface_Temperature.csv"
# __CO2_FILE_NAME:      str = "Indicator_3_2_Climate_Monthly_Atmospheric_Carbon_Dioxide_concentrations.csv"
# __SEA_LVLS_FILE_NAME: str = "Indicator_3_3_Climate_Change_In_Mean_Sea_Levels.csv"

__ABSOLUTE_PATH_TO_PROJECT_DIR: str = os.getcwd()

__RAW_DATA_DIR_NAME:      str = "raw"
__INTERIM_DATA_DIR_NAME:  str = "interim"

# --- Raw data ---

#TODO: Update this to include the relevant files for data_processor to produce the interim climate data set (currently the interim dataset is created with Excel)
# TEMP_FILE_PATH:     str = os.path.join("/", __ABSOLUTE_PATH_TO_PROJECT_DIR, "data", "climate_prediction", "raw", __TEMP_FILE_NAME)
# CO2_FILE_PATH:      str = os.path.join("/", __ABSOLUTE_PATH_TO_PROJECT_DIR, "data", "climate_prediction", "raw", __CO2_FILE_NAME)
# SEA_LVLS_FILE_PATH: str = os.path.join("/", __ABSOLUTE_PATH_TO_PROJECT_DIR, "data", "climate_prediction", "raw", __SEA_LVLS_FILE_NAME)

GLOBAL_NOAA_MEAN_ANOMALY_SEA_LEVELS_PATH: str = os.path.join(
  "/", 
  __ABSOLUTE_PATH_TO_PROJECT_DIR, 
  "data", 
  "climate_prediction", 
  __RAW_DATA_DIR_NAME, 
  "global_NOAA_mean_anomaly_sea_levels_1992_to_2025.csv"
)

# --- Interim data ---

SEA_LVL_INTERIM_DATA_FILE_NAME: str = os.path.join(
  "/", 
  __ABSOLUTE_PATH_TO_PROJECT_DIR, 
  "data", 
  "climate_prediction", 
  __INTERIM_DATA_DIR_NAME, 
  "interim_sea_level_data.csv"
)

CLIMATE_DATASET_FILENAME: str = os.path.join(
  "/", 
  __ABSOLUTE_PATH_TO_PROJECT_DIR, 
  "data", 
  "climate_prediction", 
  __INTERIM_DATA_DIR_NAME, 
  "interim_climate_dataset_v2.csv"
)

# --- Model Ouput File Names

__MODEL_OUTPUT_DIR_PATH: list [str] = [
  "/",
  __ABSOLUTE_PATH_TO_PROJECT_DIR,
  "src",
  "models",
  "climate_prediction",
  "output",
]

__GBDT_MODEL_DIR_PATH: list[str] = [
  *__MODEL_OUTPUT_DIR_PATH,
  "gbdt_model"
]

__GAM_MODEL_DIR_PATH: list[str] = [
  *__MODEL_OUTPUT_DIR_PATH,
  "gam_model"
]

GBDT_TEMPERATURE_MODEL_FILENAME: str = os.path.join(
  *__GBDT_MODEL_DIR_PATH,
  "gbdt_temperature_model.json"
)

GBDT_SEA_LEVEL_MODEL_FILENAME: str = os.path.join(
  *__GBDT_MODEL_DIR_PATH,
  "gbdt_sea_level_model.json"
)

GBDT_TEMPERATURE_MODEL_FILENAME_BIN: str = os.path.join(
  *__GBDT_MODEL_DIR_PATH,
  "gbdt_temperature_model.bin"
)

GBDT_SEA_LEVEL_MODEL_FILENAME_BIN: str = os.path.join(
  *__GBDT_MODEL_DIR_PATH,
  "gbdt_sea_level_model.bin"
)

GAM_TEMP_MODEL_FILENAME: str = os.path.join(
  *__GAM_MODEL_DIR_PATH,
  "temperature_gam_model.pkl"
)

GAM_SEA_LEVEL_MODEL_FILENAME: str = os.path.join(
  *__GAM_MODEL_DIR_PATH,
  "sea_level_gam_model.pkl"
)

GAM_VISUALIZATION_FILENAME: str = os.path.join(
  *__GAM_MODEL_DIR_PATH,
  "gam_model_performance.png"
)

# === ACTION SUGGESTION ===

# === GAME OPPONENT ===
