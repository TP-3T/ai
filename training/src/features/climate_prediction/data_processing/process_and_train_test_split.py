from features.climate_prediction.constants.climate_prediction_dataset_cols import YEAR # type: ignore
from features.climate_prediction.constants.data_file_paths import CLIMATE_DATASET_FILENAME, TESTING_CLIMATE_DATASET_FILEPATH, TRAINING_CLIMATE_DATASET_FILEPATH # type: ignore
from features.climate_prediction.constants.messages import DATASET_LOADED_MSG, LOADING_DATASET_MSG # type: ignore
from features.climate_prediction.constants.other import CSV_ENCODING # type: ignore
import pandas as pd
from pandas import DataFrame
import numpy as np

# the chronologically last 15% of the dataset is used for testing
TESTING_SET_FRACTION: float = 0.15

# the chronologically first 85% of the dataset is used for training
TRAINING_SET_FRACTION: float = 1.0 - TESTING_SET_FRACTION



def __load_interim_climate_dataset() -> DataFrame:
    """
    Load the climate dataset from CSV file.
    
    Returns:
        DataFrame: The loaded climate dataset
    """
    print(f"\n{LOADING_DATASET_MSG}...")
    
    # Load dataset - adjust path as needed for your project structure
    climate_dataset: DataFrame = pd.read_csv(CLIMATE_DATASET_FILENAME, encoding=CSV_ENCODING) # type: ignore
    
    print(f"{DATASET_LOADED_MSG}")
    print(f"Dataset shape: {climate_dataset.shape}")
    print(f"Columns: {list(climate_dataset.columns)}\n")
    
    return climate_dataset



def __remove_unused_features_excl_row_id(climate_dataset: DataFrame):
    """
    Remove usused features from the climate dataset, excluding the row id 
    (row id used for indexing but not during training, validation, or testing).
    """
    return climate_dataset.drop(columns=[YEAR])



def __create_chronologically_split_train_and_test_datasets_as_csvs(climate_dataset: DataFrame):
    """
    Chronologically split the climate dataset into training and testing sets, 
    and export the dataframes to .csv files in the 'processed' data directory.
    
    The first 85% is for training and validation, and last 15% is for testing.
    
    We are splitting chronologically because the climate dataset is time series, 
    and the model shouldn't have knowledge of the future which would happen if we did random splitting.
    """
    num_rows_in_dataset: int = len(climate_dataset)

    num_rows_in_testing_set: int = int( \
        np.floor(num_rows_in_dataset * TESTING_SET_FRACTION) \
    )
    num_rows_in_training_set: int = num_rows_in_dataset - num_rows_in_testing_set

    # * dataset already sorted by date

    # training set is from the first year up to and incl the year row at 85%
    # testing set is from after the year row at 85% up to and incl the last year%
    training_set: DataFrame = climate_dataset.iloc[:num_rows_in_training_set].copy()
    testing_set: DataFrame = climate_dataset.iloc[num_rows_in_training_set:].copy()

    # save the training and testing datasets as csv files
        # no index bc we already created one in the interim climate dataset
    training_set.to_csv(TRAINING_CLIMATE_DATASET_FILEPATH, index=False, encoding=CSV_ENCODING)
    testing_set.to_csv(TESTING_CLIMATE_DATASET_FILEPATH, index=False, encoding=CSV_ENCODING)



def process_and_train_test_split():
    
    climate_dataset: DataFrame = __load_interim_climate_dataset()
    climate_dataset_used_features: DataFrame = __remove_unused_features_excl_row_id(climate_dataset)
    # (climate_dataset_features, climate_dataset_targets) = __separate_features_and_targets(climate_dataset)

    #  For both the features and targets 
    __create_chronologically_split_train_and_test_datasets_as_csvs(climate_dataset_used_features)



if __name__ == "__main__":
    process_and_train_test_split()
