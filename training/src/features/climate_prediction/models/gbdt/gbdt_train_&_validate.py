"""
Second iteration of climate prediction model training using the gradient boosted decision tree algorithm.
Contains functions to visualize the climate data, as well as perform the model training, walk-forward validation and testing.
"""

import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


# === Load climate prediction dataset ===


# === Chronologically split the dataset into a training and testing sets ===

# (first 85% is for training and validation, last 15% is for testing)
    # (we are splitting chronologically because the climate dataset is time series, and the model shouldn't have knowledge of the future which would happen by randomly splitting)

# === Determine the most optimal combination of gbdt hyperparameters ===

# Perform multiple iterations of training and validaton - using a time-series aware cross validation technique called walk-forward validation

# --- Walk-forward validation ---

  # - Multiple iterations of training and then validating a model, 
  #   and each validation process produces a root mean squared error (RMSE)

  # - After the total number of iterations are complete, all of the RMSE's are averaged

  # - This allows you to get a more robust and reliable indicator of model performance,
  #   compared to a single validation set, because it eliminates the problem of the model 
  #   being limited to training on just a certain section of the data

      # Basically it ensures that we try training and validating that type of model on all rows of the data AT LEAST ONCE 
      # (not the same model, but is the same type of model), 
      # compared to a single train validation set split, where
      # the type of model only gets trained on only a specific subset of the data, and validated on the other section

  # Num model "instances" trained == num iterations == number of times the model type is retrained
  # (each model has the same hyperparameter combination)

  # - In each iteration, a different "instance" of the same type of model is trained and validated on a different proportion of the initial 85% dataset (NOT INCLUDING THE TESTING SUB-DATASET)
  # - In each iteration, a proportion of the 85% dataset is taken (a different proportion for each iteration).
  # - That proportion is then split Chronologically into train and validation sets, 
  #   with the past data being used for training, and future data being used for validation

  # - The validation process in each iteration produces a RMSE
  # - After all iterations, the RMSE's are averaged together, 
  #   and that average RMSE is the final model performance metric

# === Select hyperparameters based which combination produced the "best" mean validation scores ===

# === "Retrain" the final model on all pre-test data ===
