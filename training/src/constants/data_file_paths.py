"""Constants file for raw dataset file paths."""

import os

# === CLIMATE PREDICTION ===
# __TEMP_FILE_NAME:     str = "Indicator_3_1_Climate_Annual_Mean_Global_Surface_Temperature.csv"
# __CO2_FILE_NAME:      str = "Indicator_3_2_Climate_Monthly_Atmospheric_Carbon_Dioxide_concentrations.csv"
# __SEA_LVLS_FILE_NAME: str = "Indicator_3_3_Climate_Change_In_Mean_Sea_Levels.csv"

__ABSOLUTE_PATH_TO_PROJECT_DIR: str = os.getcwd()

__RAW_DATA_DIR_NAME:      str = "raw"
__INTERIM_DATA_DIR_NAME:  str = "interim"

# --- Raw data ---

#TODO: Update this to include the relevant files for data_processor to produce the interim climate data set (currently the interim dataset is generated using Excel)
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
  "interim_climate_dataset.csv"
)

# === ACTION SUGGESTION ===

# === GAME OPPONENT ===
