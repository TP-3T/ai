
"""
Main file of climate change data processor.

Has functions to:

Preprocesses the required raw data files, and creates a dataset file with annual frequency and using global averages, consisting of the features:
 - Year
 - CO2 Concentration (average CO2 concentration (ppm) - temperature change with respect to a baseline climatology, corresponding to the period 1951-1980
 - Temperature (average global temperature) (degree Celsius) - calculated from average change, relative to 1951-1980 average (14.0 deg C)
 - Sea Level (average change in global mean) (mm) - via NASA TOPEX/Poseidon

Split the data into training, validation, and training sets, using sklearn train_test_split function.
"""

# import numpy as np
# import pandas as pd
# from pandas import Series # type: ignore
# from pandas import DataFrame # type: ignore

# from data_processing.constants.data_file_paths import TEMP_FILE_PATH # type: ignore
# from data_processing.constants.data_file_paths import CO2_FILE_PATH # type: ignore
# from data_processing.constants.data_file_paths import SEA_LVLS_FILE_PATH # type: ignore

# from data_processing.constants.climate_dataset_columns import TEMP_FIRST_YEAR_DATA_COLLECTION # type: ignore
# from data_processing.constants.climate_dataset_columns import TEMP_LAST_YEAR_DATA_COLLECTION # type: ignore


#TODO: Update functions to process the required datasets and extract relevant features the interim climate data set (currently the interim dataset is generated using Excel)

# def load_and_process_temperature_dataset() -> Series[float]:
#   temp_df = pd.read_csv(TEMP_FILE_PATH)  # type: ignore

#   # temp_unit:      str = str(temp_df.at[0, "Unit"])
#   # temp_indicator: str = str(temp_df.at[0, "Indicator"])

#   # Get all of the rows (:), and the columns from the first to last year of data collection
#   country_temp_changes_by_year: DataFrame = temp_df.loc[:, str(TEMP_FIRST_YEAR_DATA_COLLECTION) : str(TEMP_LAST_YEAR_DATA_COLLECTION)]

#   row_axis_num: int = 0

#   avg_global_temp_change_each_year: Series[float] = country_temp_changes_by_year.mean(axis=row_axis_num, skipna=True)

#   # for i in range(0, len(avg_global_temp_change_each_year)):
#   #   print(f"{i}: {avg_global_temp_change_each_year[i]}")
#   return avg_global_temp_change_each_year

# def load_and_process_co2_dataset():# -> Series[float]
#   co2_df = pd.read_csv(CO2_FILE_PATH)  # type: ignore

# def load_and_process_sea_lvl_dataset():# -> Series[float]
#   sea_lvl_df = pd.read_csv(SEA_LVLS_FILE_PATH)  # type: ignore

#TODO: Split the data into training, validation, and training sets, using sklearn train_test_split function.

# if __name__ == "__main__":
#   load_and_process_temperature_dataset()
