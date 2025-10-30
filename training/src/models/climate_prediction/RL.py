"""
Climate prediction model training using reinforcement learning.
"""


import numpy as np
import pandas as pd
import gymnasium as gym #OPENAI library for RL environments
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import r2_score
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO #RL algorithm

#adjust depending on where you are running from 
from src.constants.data_file_paths import CLIMATE_DATASET_FILENAME # type: ignore

YEAR: str = "Year"
CO2: str = "Global Average CO2 Concentration (ppm)"
TEMP: str = "Global Average Temperature (deg C)"
SEA_LVL: str = "Global Average Absolute Sea Level (mm)"

COL_AXIS_NUM = 1

class ClimateEnv(gym.Env):
  """
  Custom Environment for climate prediction using reinforcement learning.
  """
  def __init__(self, climate_dataset: DataFrame):
    super(ClimateEnv, self).__init__()

    self.climate_dataset = climate_dataset
    self.features = climate_dataset[[YEAR, CO2, TEMP, SEA_LVL]].values
    self.targets = climate_dataset[[CO2, TEMP, SEA_LVL]].values

    # Define action and observation space
    # Actions: Adjust predictions for CO2, Temperature, and Sea Level
    self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    # Observations: Current state (features)
    self.observation_space = spaces.Box(
        low=np.min(self.features, axis=0),
        high=np.max(self.features, axis=0),
        shape=self.features.shape[1:],
        dtype=np.float32,
    )

    self.current_step = 0
    self.predictions = np.mean(self.targets, axis=0) #Initial predictions

  def reset(self, seed=None, options=None):
    """
    Reset the environment to the initial state.
    """
    super().reset(seed=seed)
    self.current_step = 0
    self.predictions = np.mean(self.targets, axis=0) #Reset predictions to initial mean values
    return self.features[self.current_step], {}
  
  def step(self, action):
    """
    Take an action in the environment.
    """
    self.predictions += action 

    actual = self.targets[self.current_step]
    reward = -np.sum(np.abs(actual - self.predictions))

    self.current_step += 1
    done = self.current_step >= len(self.features)
    truncated = False

    next_state = self.features[self.current_step] if not done else self.features[-1]

    return next_state, reward, done, truncated, {}
  
def train_rl_agent(env: ClimateEnv, timesteps: int = 10000) -> PPO:
  """
  Train a reinforcement learning agent using PPO algorithm.
  """
  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=timesteps)
  return model

def evaluate_rl_agent(env: ClimateEnv, model):
  """
  Evaluate the trained RL agent.
  """
  obs, info = env.reset()
  predictions = []
  actuals = []

  while True:
    action, _ = model.predict(obs)
    obs, _, done, truncated, _ = env.step(action)
    predictions.append(env.predictions.copy())
    actuals.append(env.targets[env.current_step - 1])

    if done or truncated:
      break

  predictions = np.array(predictions)
  actuals = np.array(actuals)

  for i, target_name in enumerate([CO2, TEMP, SEA_LVL]):
    rmse = root_mean_squared_error(actuals[:, i], predictions[:, i])
    r2 = r2_score(actuals[:, i], predictions[:, i])
    print(f"{target_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")




def __load_climate_dataset() -> DataFrame:
  # Load climate dataset
  climate_dataset: DataFrame = pd.read_csv(CLIMATE_DATASET_FILENAME) # type: ignore
  return climate_dataset



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

  train_data, test_data = train_test_split(climate_dataset, test_size=0.25, random_state = 1)
  env = ClimateEnv(train_data)
  rl_agent = train_rl_agent(env)

  test_env = ClimateEnv(test_data)
  evaluate_rl_agent(test_env, rl_agent)

  visualize_dataset(climate_dataset)