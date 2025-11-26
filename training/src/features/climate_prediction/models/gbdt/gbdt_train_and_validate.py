"""
Second iteration of climate prediction model training using the gradient boosted decision tree algorithm.
Contains a function to train the model with walk-forward validation, that exports the resulting model file to a csv.
"""

from features.climate_prediction.constants.data_file_paths import TRAINING_CLIMATE_DATASET_FILEPATH # type: ignore
from features.climate_prediction.data_processing.separate_features_and_targets import separate_features_and_targets_from_climate_dataset # type: ignore
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# as many as needed for fastest training in parallel
NUM_DEVICE_CPU_CORES: int = -1

VERBOSITY_OF_VALIDATON_LOGS: int = 1

ML_MODEL_TECHNIQUE_NAME: str = "xgb"
RANDOM_STATE_VALUE: int = 42

OBJECTIVE_FUNCTION: str = "reg:squarederror" # objective function that optimizes the Mean Squared Error (MSE) loss function

# Lists of potential hyperparameters for training 
# (the most optimal combination of these will be determined using grid search)
LEARNING_RATE:      list[float] = [0.03, 0.05, 0.1]
NUM_BOOST_ROUNDS:   list[int] = [50, 100, 150, 200] # same as number of trees
MAX_TREE_DEPTH:     list[int] = [3, 4, 5]
MIN_CHILD_WEIGHT:   list[int] = [1, 3, 5]
SUBSAMPLING_RATE:   list[float] = [0.8, 1.0]
FEAT_SAMPLING_RATE: list[float] = [0.8, 1.0]



def train_and_validate_gbdt_model() -> Pipeline:
  """
  Train and validate gbdt model on the training set
  using grid search with time series aware cross validation 
  (called walk-forward validation).
  """
  # === Load the training climate prediction dataset ===
  training_climate_dataset: DataFrame = pd.read_csv(TRAINING_CLIMATE_DATASET_FILEPATH) # type: ignore

  # === Seperate features and targets from training dataset ===
  (climate_dataset_features_train, \
   climate_dataset_targets_train) = separate_features_and_targets_from_climate_dataset(training_climate_dataset)
  
  # Instantiate gbdt model
  xgb_model: XGBRegressor = XGBRegressor(
    objective=OBJECTIVE_FUNCTION,
    tree_method="hist",
    eval_metric="rmse",
    random_state=RANDOM_STATE_VALUE,
  )

  # Hyperparams
  param_grid: dict[str, str] = { # type: ignore
    "xgb__n_estimators": NUM_BOOST_ROUNDS,
    "xgb__learning_rate": LEARNING_RATE,
    "xgb__max_depth": MAX_TREE_DEPTH,
    "xgb__min_child_weight": MIN_CHILD_WEIGHT,
    "xgb__subsample": SUBSAMPLING_RATE,
    "xgb__colsample_bytree": FEAT_SAMPLING_RATE,
  }

  # A Pipeline allows for assembling 
  # a series steps that are done automatically using sci-kit learn methods 
  # (passed into the steps array)
  ml_pipeline = Pipeline(
    steps=[
      (ML_MODEL_TECHNIQUE_NAME, xgb_model)
    ]
  )

  # Perform multiple iterations of training and validaton - using a time-series aware cross validation technique called walk-forward validation

  # --- Walk-forward validation (using time series cross-validator) ---

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

  # num_splits == number of iterations of training & validation, each iteration training on different split of data
  num_splits: int = 5
  time_series_cross_validator: TimeSeriesSplit = TimeSeriesSplit(n_splits=num_splits)

  # === 
  # Use Grid Search cross validaton method to determine the 
  # most optimal hyperparameter combination for gbdt training 
  # ===

  # Perform grid search, which will:
  # - Run MULTIPLE ITERATIONS of walk-forward validation (one for EVERY combination of model hyperparameters),
  #   which the result of this is a list of average RMSEs
  grid_search: GridSearchCV = GridSearchCV(
    estimator=ml_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error", # negative RMSE, higher is better
    cv=time_series_cross_validator,
    n_jobs=NUM_DEVICE_CPU_CORES,
    verbose=VERBOSITY_OF_VALIDATON_LOGS,
    refit=True # retrain the model that was trained with the most optimal hyperparameters on the full training set
  )

  # Train
  grid_search.fit( # type: ignore
    climate_dataset_features_train, 
    climate_dataset_targets_train
  )

  # === Select the hyperparameters combination that produced the "best" average mean validation scores ===

    # - Using the resulting list of average RMSE's, 
    #   it chooses the combination of hyperparams that resulted in the lowest RMSE value
    #   and sets the best model after refitting it on the entire training dataset

  gbdt_model_optimal_hyperparams: Pipeline = grid_search.best_estimator_

  return gbdt_model_optimal_hyperparams
