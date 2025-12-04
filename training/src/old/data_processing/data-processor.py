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
import pandas as pd
from pandas import Series # type: ignore
from pandas import DataFrame # type: ignore

# from data_processing.constants.data_file_paths import TEMP_FILE_PATH # type: ignore
# from data_processing.constants.data_file_paths import CO2_FILE_PATH # type: ignore
# from data_processing.constants.data_file_paths import SEA_LVLS_FILE_PATH # type: ignore

# from data_processing.constants.climate_dataset_columns import TEMP_FIRST_YEAR_DATA_COLLECTION # type: ignore
# from data_processing.constants.climate_dataset_columns import TEMP_LAST_YEAR_DATA_COLLECTION # type: ignore
from constants.data_file_paths import GLOBAL_NOAA_MEAN_ANOMALY_SEA_LEVELS_PATH, SEA_LVL_INTERIM_DATA_FILE_NAME # type: ignore

ROW_AXIS_NUM = 0
COL_AXIS_NUM = 1

#TODO: Update functions to process the required datasets and extract relevant features the interim climate data set (currently the interim dataset is created manually using Excel)

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

def load_data_and_calculate_remaining_sea_level_change_vals():
  """
  Create a dataset containing the remaining changes in sea level for the years 2016 - 2024, 
  from the global NOAA mean anomaly sea levels dataset that covers the years 1992 to 2025.

  This dataset is to be used for use the interim climate dataset change in sea level column.
  """
  VALID_YEARS = "2015.|2016.|2017.|2018.|2019.|2020.|2021.|2022.|2023.|2024."
  YEAR_COL_NAME: str = "Year"
  YEAR_GROUP_COL_NAME: str = "Year_Truncated"
  MEAN_SEA_LEVEL_ANOMALY_COL_NAME: str = "Mean Sea Level Anomaly" 

  mean_sea_lvl_anomaly_df: DataFrame = pd.read_csv(GLOBAL_NOAA_MEAN_ANOMALY_SEA_LEVELS_PATH)  # type: ignore

  # Remove all of the rows except 2015 - 2024
  
    # Select all rows in the column "Year"
  years:            Series[float] = mean_sea_lvl_anomaly_df.loc[:, YEAR_COL_NAME]
  years_str:        Series[str] = years.astype(str)
  years_truncated:  Series[int] = years.astype(int)

    # Filter out the rows where the year does not include any of the years 2015 - 2024
  mean_sea_lvl_anomaly_filtered_yrs_df: DataFrame = mean_sea_lvl_anomaly_df.loc[years_str.str.contains(VALID_YEARS, case=False, na=False), :] # keep all columns but only rows that pass the condition

  # Get the last non empty column cell in each row (the column in the row being the sea lvl anomaly measurement method)
  single_mean_sea_level_df: Series[float] = mean_sea_lvl_anomaly_filtered_yrs_df.apply(get_last_non_NaN_from_row, axis=COL_AXIS_NUM)

  # Create new dataframe with the year and the mean anomaly sea level
  single_mean_sea_lvl_by_sub_year_df: DataFrame = pd.DataFrame({
    YEAR_COL_NAME: mean_sea_lvl_anomaly_filtered_yrs_df.loc[:, YEAR_COL_NAME],
    MEAN_SEA_LEVEL_ANOMALY_COL_NAME: single_mean_sea_level_df
  })
  
  # Create DF copy to avoid SettingWithCopyWarning error
  sea_lvl_copy_df: DataFrame = single_mean_sea_lvl_by_sub_year_df.copy()

  # Add the column with trucated year value to the existing dataframe
  sea_lvl_copy_df[YEAR_GROUP_COL_NAME] = years_truncated.astype(str)

  # Group mean sea levels by year (one row per year) and average each group to produce annual averages
  annual_sea_lvl_anomaly_df: DataFrame = sea_lvl_copy_df.groupby(YEAR_GROUP_COL_NAME).mean() # type: ignore

  # Drop the subyear column
  annual_sea_lvl_anomaly_df: DataFrame = annual_sea_lvl_anomaly_df.drop(columns=[YEAR_COL_NAME])

  annual_change_in_sea_lvl_df: DataFrame = calc_sea_level_change_from_anomaly(annual_sea_lvl_anomaly_df, MEAN_SEA_LEVEL_ANOMALY_COL_NAME)

  return annual_change_in_sea_lvl_df

def get_last_non_NaN_from_row(row: Series) -> float:
  """
  Get the last valid value in a series.
  """
  return row[row.last_valid_index()]



def calc_sea_level_change_from_anomaly(
    annual_sea_lvl_anomaly_df: DataFrame, 
    sea_lvl_anomaly_col_name: str
) -> DataFrame:
  """
  Calculate the annual changes in sea level from the annual mean anomaly values, using the formula:
  Change in sea level = A_t - A_(t-1).
  """
  annual_change_in_sea_lvl_df: DataFrame = annual_sea_lvl_anomaly_df.copy()

  # df.diff() subtracts the current minus previous for each element in the DataFrame or Series
  annual_change_in_sea_lvl_df.loc[:, sea_lvl_anomaly_col_name] = annual_sea_lvl_anomaly_df.loc[:, sea_lvl_anomaly_col_name].diff()

  # Drop the first NaN row (year 2015)
  annual_change_in_sea_lvl_df = annual_change_in_sea_lvl_df.dropna()  # type: ignore

  return annual_change_in_sea_lvl_df

def process_required_sea_level_data_and_create_csv():
  annual_change_in_sea_lvl_df: DataFrame = load_data_and_calculate_remaining_sea_level_change_vals()
  #TODO: set in interim climate dataset file
  annual_change_in_sea_lvl_df.to_csv(SEA_LVL_INTERIM_DATA_FILE_NAME) # type: ignore

if __name__ == "__main__":
  """
  Preprocess climate related data and output to climate-prediction/interim directory.
  """
  process_required_sea_level_data_and_create_csv()
  # load_and_process_temperature_dataset()
