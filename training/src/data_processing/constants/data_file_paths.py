"""Constants file for raw dataset file paths."""

import os

# === CLIMATE PREDICTION ===
__TEMP_FILE_NAME:     str = "Indicator_3_1_Climate_Annual_Mean_Global_Surface_Temperature.csv"
__CO2_FILE_NAME:      str = "Indicator_3_2_Climate_Monthly_Atmospheric_Carbon_Dioxide_concentrations.csv"
__SEA_LVLS_FILE_NAME: str = "Indicator_3_3_Climate_Change_In_Mean_Sea_Levels.csv"

__ABSOLUTE_PATH_TO_PROJECT_DIR: str = os.getcwd()

#TODO: Update this to include the relevant files for data_processor to produce the interim climate data set (currently the interim dataset is generated using Excel)
TEMP_FILE_PATH:     str = os.path.join("/", __ABSOLUTE_PATH_TO_PROJECT_DIR, "data", "climate_prediction", "raw", __TEMP_FILE_NAME)
CO2_FILE_PATH:      str = os.path.join("/", __ABSOLUTE_PATH_TO_PROJECT_DIR, "data", "climate_prediction", "raw", __CO2_FILE_NAME)
SEA_LVLS_FILE_PATH: str = os.path.join("/", __ABSOLUTE_PATH_TO_PROJECT_DIR, "data", "climate_prediction", "raw", __SEA_LVLS_FILE_NAME)

# === ACTION SUGGESTION ===

# === GAME OPPONENT ===
