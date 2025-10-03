import os

__RAW_CLIMATE_DATA_FILE_DIR_PATH: str = "data/climate_prediction/raw"

TEMP_FILE_NAME: str     = "Indicator_3_1_Climate_Annual_Mean_Global_Surface_Temperature"
CO2_FILE_NAME: str      = "Indicator_3_2_Climate_Monthly_Atmospheric_Carbon_Dioxide_concentrations"
SEA_LVLS_FILE_NAME: str = "Indicator_3_3_Climate_Change_In_Mean_Sea_Levels"

__ABSOLUTE_PATH_TO_PROJECT_DIR: str = os.getcwd()

TEMP_FILE_PATH: str = os.path.join(TEMP_FILE_NAME, "/", __ABSOLUTE_PATH_TO_PROJECT_DIR, __RAW_CLIMATE_DATA_FILE_DIR_PATH)

print(TEMP_FILE_PATH)
