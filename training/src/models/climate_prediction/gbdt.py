"""
Climate prediction model training using the gradient boosted decision tree algorithm.
Contains functions to visualize the climate data, as well as perform the model training, validation and testing.
"""

# Machine Learning (for supervised learning and prediction):
#   - Training data used is a set of multiple features (X) and one target (Y) (inputs and expected output)
#   - Each row in a dataset with the features X and the target Y make up one data point
#   - Overview: A machine learning model learns from training data points to generalize for unseen ones
#     - The model in training makes a prediction of what the target should be
#     - After making a prediction of the target, the model compares its prediction with the expected target value
#     - Based on the amount of error it made, it adjusts its "parameters" (basically the model's setting values) to hopefully make a better prediction on the next data point
#     - The final parameter values after training is what makes up the machine learning model's ability to predict unforeseen data
#     - Goal is to improve prediction accuracy and minimize the amount of error, by optimizing the model parameters

# Steps to optimize model parameters:
#   1. Calculate and quantify the amount of error (using the loss function)
#   2. Compute the slope of a line that "touches" the loss function at a point - called the gradient 
#   3. Update the model parameters according to the gradient to minimize loss

# Loss function:
#   - A function that quantifies the difference between a model's predictions and actual values
#   - Used to:
#       - Calculate the error (predicted output compared to actual value)
#       - Quantify the amount of error to be able to update parameter values throughout model training
#       - Evaluate the model (compare loss on training, validation, and test datasets)
#   - The same loss function is used to evaluate each prediction during the training process

# Mean Squared Error:
#   - A type of loss function commonly used for regression
#   - Is the sum of squared differences between predicted and actual values
#       - Error (for one data point) = 1/2 x (Observed - Predicted)^2
#   - Purpose of squaring is to prevent negative errors, and emphasize larger errors for the model to correct
#   - Often the error is multiplied by 1/2:     Error = 1/2 x (Observed - Predicted)^2
#   - Purpose of x 1/2 is to produce a simple equation after differentiation of the loss function - result will just be: ( Observed - Predicted )
#   - Differentiate - finding the slope of a line that "touches" the loss function curve at a point
#   - Differentiation is done to produce a value that can be used to adjust model parameters (an algorithm uses the value to calculate the adjustment amounts)

# Train/Test Split and Generalization:

# Bias-Variance Tradeoff:

# Gradient Boosting:
#   - An ensemble technique in ML
#   - Does not learn from data independently
#   - It instead combines the predictions of multiple weak learners (models that performs slightly better than randomly guessing)
#     into one more accurate strong learner
#       - An example of weak learner is a decision tree
#   - The strong learner is a combination of many weak learners and is used for the same set of features and target combination (different data points though)

# Gradient boosting algorithm:
#   - Works for tabular data
#   - Supervised learning technique
#   - Can be used for both regression (predicting continous, numerical values) or classification (predicting discrete categories for data - each category being a class label)
#   - Each weak learner is trained using all of the rows of the training data
#   - Each tree uses the features and prediction error, to predict what corrections are needed to be able to 
#   - Trees don’t predict the final answers. Instead they predict how to correct the errors of previous predictions

# Gradient boosted decision tree:
#   - The Random Forest (RF) technique in ML is an ensemble of decision trees, where each tree is built independently and in parallel.
#       - Then it combines the prediction results at the end
#   - Unlike RF, the Gradient boosted decision tree technique builds trees sequentially.
#       - Because of this, each tree learns from the prediction error of the previous trees in the ensemble, and uses the error when making its own prediction
#   - Uses gradient boosting algorithm

# =============================================================================

# Gradient Boosting Algorithm Steps:
 
# 1. Make an initial prediction
#       - Often simply the mean of the target values.
#       - This gives a baseline model for all data points.

# 2. Calculate the pseudo-residuals
#       - Pseudo-residual = Observed value - Current prediction
#       - These represent how much correction each data point needs.

# 3. Build a weak learner (decision tree) that will predict these pseudo-residuals (for indirectly predicting the target vlaue)
#       - Each split from a decision tree node is based on features (ex. CO2 < 301.2)
#       - Each leaf node should be an error correction amount to add to the initial prediction, which results in the predicted target value

# 4. Update the prediction
#       - New prediction of the target = Initial prediction + (learning_rate * tree_leaf_value)

# 5. Repeat steps 2-4 with updated pseudo-residuals
#       - Each new tree fixes errors made by the previous trees.
#       - The model gradually improves by correcting mistakes.

# =============================================================================

# Hyperparameters - model "settings":
# - Properties of the ensemble DTs and gradient boosting algorithm that are set before training
# - Determined and "tuned" by ML trainer
#     - Objective - the loss function to use for the algorithm
#     - Learning Rate - controls the contribution of each DT - adjusts the shrinkage factor 
#           - Ex. smaller values means each DT prediction has lower contribution to the final result in the ensemble
#     - Number of trees - number of boosting rounds = number of trees to build in the ensemble
#     - Max tree depth - the number of levels in each DT (including the leaf level)
#     - Minimum number of samples per leaf (or min child weight) - minimum sum of instance weights required in a leaf. Prevents a tree from creating leaves with very few samples.
#     - Subsampling rate - proportion of rows used to train each tree 
#           - Ex. Each DT is trained on a randomly sampled 70% of the rows (a random 70% of the rows, not the same 70% for every DT)
#     - Feature sampling rate - proportion of columns (features) used to train each tree
#           - Ex. Each DT is trained with a randomly sampled 50% of the features (a random 50% of the columns, not the same 50% for every DT)

# objective - Mean Squared Error (MSE) loss function
# learning rate - 
# # trees - early stopping
# max tree depth - 3
# min child weight - 3
# subsampling rate - 0.8
# feature sampling rate - 1.0 (only have two features)

# will use xgboost since sklearn only supports training using CPU resources

# =============================================================================

import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import r2_score
import xgboost as xgb
from xgboost import Booster, DMatrix
import matplotlib.pyplot as plt

from constants.data_file_paths import CLIMATE_DATASET_FILENAME # type: ignore

YEAR: str = "Year"
CO2: str = "Global Average CO2 Concentration (ppm)"
TEMP: str = "Global Average Temperature (deg C)"
SEA_LVL: str = "Global Average Absolute Sea Level (mm)"

COL_AXIS_NUM = 1

def __load_climate_dataset() -> DataFrame:
  # Load climate dataset
  climate_dataset: DataFrame = pd.read_csv(CLIMATE_DATASET_FILENAME) # type: ignore
  return climate_dataset

def train_validate_and_test_model(climate_dataset: DataFrame):
  """
  Train, validate, and test a regression based ML model for climate prediction. ML technique used for training is the gbdt algorithm.
  The package xgboost is used for the implementation of the above. 
  Dataset splitting and model evaluation is implemented using the sklearn package.
  """
  TEST_PREDICTION_RESULTS_MSG:  str = "Testing Dataset Prediction Results"
  ACTUAL_SEA_LVL:               str = "Actual Sea level"
  PREDICTED_SEA_LVL:            str = "Predicted Sea level"
  RMSE_REPORT_MSG:              str = "RMSE (average error) of the model (units - mm)"
  COEFF_OF_DET_REPORT_MSG:      str = "Model goodness of fit"

  # Hyperparameters for training
  OBJECTIVE:          str = "reg:squarederror" # use Mean Squared Error (MSE) loss function
  LEARNING_RATE:      float = 0.1
  NUM_BOOST_ROUNDS:   int = 100 # same as number of trees
  MAX_TREE_DEPTH:     int = 3
  MIN_CHILD_WEIGHT:   int = 5
  SUBSAMPLING_RATE:   float = 0.8
  FEAT_SAMPLING_RATE: float = 1.0
  TREE_METHOD_SPLIT_ALGORITHM: str = "exact" # what algorithm to use for constructing the individual decision trees - explained in section 3 in the xgboost article explanation

  DEVICE: str = "gpu"

  # Extract required features and target
  # Features: Global avg temperature, global avg CO2
  # Target:   Global absolute avg sea level
  features: DataFrame = climate_dataset.drop(columns=[YEAR, SEA_LVL], axis=COL_AXIS_NUM)
  target:   DataFrame = climate_dataset.filter(like=SEA_LVL, axis=COL_AXIS_NUM)

  # Split the data into training and testing sets (default - 0.75 train size, 0.25 test size) 
  # Reproducible split of data (not random)
  feat_train, feat_test, target_train, target_test = train_test_split(features, target, random_state=1) # type: ignore

  dtrain_regr: DMatrix = xgb.DMatrix(feat_train, target_train)
  dtest_regr:  DMatrix = xgb.DMatrix(feat_test, target_test)

  params = { # type: ignore
    "device": DEVICE, # use gpu of device for training (will switch to cpu if running device has no compatible gpu)
    "tree_method": TREE_METHOD_SPLIT_ALGORITHM,
    "objective": OBJECTIVE,
    'eta': LEARNING_RATE,
    'max_depth': MAX_TREE_DEPTH,
    'min_child_weight': MIN_CHILD_WEIGHT,
    'subsample': SUBSAMPLING_RATE,
    'colsample_bytree': FEAT_SAMPLING_RATE,
  }

  # Validation sets to view the model error amount at different points before training ends
  evals = [(dtrain_regr, "train"), (dtest_regr, "validation")]

  # Train model
  model: Booster = xgb.train(
    params=params,
    dtrain=dtrain_regr,
    num_boost_round=NUM_BOOST_ROUNDS,
    evals=evals
  )

  # Test model on unseen data
  target_test_predictions = model.predict(dtest_regr)

  # Create new dataframe with the columsn being of the test features, actual target, and predicted target
  test_features_and_target_df: DataFrame = feat_test.copy()

    # Add the actual and predicted cols
  test_features_and_target_df[ACTUAL_SEA_LVL] = target_test
  test_features_and_target_df[PREDICTED_SEA_LVL] = target_test_predictions

  print(f"\n{TEST_PREDICTION_RESULTS_MSG}:\n")
  print(test_features_and_target_df)

  # Compare predictions with the actual data by calculating the average error - using root mean squared error
  # units of rmse are in mm
  rmse: float = root_mean_squared_error(target_test, target_test_predictions) # type: ignore

  # Calculate the "goodness of the fit" / coeff of determination of the model (R^2)
  # (how close the model's predicted values are to the actual)
  r2: float = r2_score(target_test, target_test_predictions) # type: ignore

  print(f"{RMSE_REPORT_MSG}: {rmse:.5f}")
  print(f"{COEFF_OF_DET_REPORT_MSG}: {r2:.5f}")



def visualize_dataset(climate_dataset: DataFrame):
  """
  Visualize the climate dataset (see the output for the different plots).
  """

  year_col: Series = climate_dataset.loc[:, YEAR]
  temp_col: Series = climate_dataset.loc[:, TEMP]
  sea_lvl_col: Series = climate_dataset.loc[:, SEA_LVL]
  co2_col: Series = climate_dataset.loc[:, CO2]

  # Create a figure with a 2x2 grid of subplots
  # (figsize adjusts the figure size)
  fig, axs = plt.subplots(4, 4, figsize=(10, 8)) # type: ignore

  # Plot on row 0 col 0
  axs[0, 0].plot(year_col, temp_col, color='blue')
  axs[0, 0].set_title(f"{YEAR} vs {TEMP}")
  axs[0, 0].set_xlabel(YEAR)
  axs[0, 0].set_ylabel(TEMP)

  # Plot on row 0 col 1
  axs[0, 1].plot(year_col, sea_lvl_col, color='red')
  axs[0, 1].set_title(f"{YEAR} vs {SEA_LVL}")
  axs[0, 1].set_xlabel(YEAR)
  axs[0, 1].set_ylabel(SEA_LVL)

  # Plot on row 0 col 2
  axs[0, 2].plot(year_col, co2_col, color='green')
  axs[0, 2].set_title(f"{YEAR} vs {CO2}")
  axs[0, 2].set_xlabel(YEAR)
  axs[0, 2].set_ylabel(CO2)

  #  ------------------------

  # Plot on row 1 col 0
  axs[1, 0].plot(co2_col, temp_col, color='blue')
  axs[1, 0].set_title(f"{CO2} vs {TEMP}")
  axs[1, 0].set_xlabel(CO2)
  axs[1, 0].set_ylabel(TEMP)

  # Plot onrow 2 col 0
  axs[2, 0].plot(co2_col, sea_lvl_col, color='red')
  axs[2, 0].set_title(f"{CO2} vs {SEA_LVL}")
  axs[2, 0].set_xlabel(CO2)
  axs[2, 0].set_ylabel(SEA_LVL)

  # # Plot on row 3 col 0
  axs[3, 0].plot(temp_col, sea_lvl_col, color='green')
  axs[3, 0].set_title(f"{TEMP} vs {SEA_LVL}")
  axs[3, 0].set_xlabel(TEMP)
  axs[3, 0].set_ylabel(SEA_LVL)

  # Adjust layout to prevent overlapping titles/labels
  plt.tight_layout()

  # Display the figure with all the subplots
  plt.show() # type: ignore

if __name__ == "__main__":
  climate_dataset: DataFrame = __load_climate_dataset()
  train_validate_and_test_model(climate_dataset)
  visualize_dataset(climate_dataset)
