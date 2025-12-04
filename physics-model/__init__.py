"""Main file."""

from calculate_temperature import calculate_new_temperature 
from calculate_radiative_forcing import calculate_radiative_forcing
# from calculate_sea_level import calculate_new_sea_level
from calculate_sea_level_kyle import calculate_sea_level_rise

__NUM_SECONDS_IN_MIN:   float = 60.0
__NUM_MINS_IN_HOUR:     float = 60.0
__NUM_HOURS_IN_DAY:     float = 24.0
__NUM_DAYS_IN_YEAR:     float = 365.0
__NUM_SECONDS_IN_YEAR:  float = __NUM_SECONDS_IN_MIN * __NUM_MINS_IN_HOUR * __NUM_HOURS_IN_DAY * __NUM_DAYS_IN_YEAR

__CELCIUS_KELVIN_CONVERSION_VALUE: float = 273.15


def calculate_future_values(current_co2_concentration_ppm: float, current_temperature_celcius: float, current_sea_level: float, change_in_time_years: float):
  """
  Determine the future value of temperature (celcius) and sea level (mm) from the current global average temperature (celcius) and CO2 concentration (ppm), as well as the amount of time elapsed in years.
  """

  # convert celcius to Kelvin
  current_temperature_kelvin: float = current_temperature_celcius + __CELCIUS_KELVIN_CONVERSION_VALUE

  # time interval, currently one year expressed in seconds
  change_in_time_seconds: float = change_in_time_years * __NUM_SECONDS_IN_YEAR

  co2_radiative_forcing:  float = calculate_radiative_forcing(current_co2_concentration_ppm)

  future_temperature_kelvin: float = calculate_new_temperature(
    current_temperature_kelvin=current_temperature_kelvin,
    change_in_time_seconds=change_in_time_seconds,
    co2_radiative_forcing_watts_per_metre_sq=co2_radiative_forcing
  )

  # future_sea_level_mm: float = calculate_new_sea_level(
  #       current_sea_level_mm=current_sea_level_mm,
  #       current_temperature_celcius=current_temperature_celcius,
  #       change_in_time_years=change_in_time_years
  # )

  # convert Kelvin to celcius 
  future_temperature_celcius: float = future_temperature_kelvin - __CELCIUS_KELVIN_CONVERSION_VALUE

  # calculate sea level rise based on the future temperature
  sea_level_rise_mm: float = calculate_sea_level_rise(
    current_temperature_celcius=future_temperature_celcius,
    time_elapsed_years=change_in_time_years
  )
  
  future_sea_level_mm: float = current_sea_level + sea_level_rise_mm
  
  return future_temperature_celcius, future_sea_level_mm

  # return {
  #       'future_temperature_celcius': future_temperature_celcius,
  #       'future_sea_level_mm': future_sea_level_mm,
  #       'temperature_change': future_temperature_celcius - current_temperature_celcius,
  #       'sea_level_change': future_sea_level_mm - current_sea_level_mm
  # }

def main():
  """Main driver function."""

  current_CO2_ppm:                      float = 316.18 # Current CO2 concentration in ppm TODO: Update with game state's CO2 level
  current_temperature_celcius:          float = 14.5 # current temperature in Celsius TODO: Update with game state's current temperature
  change_in_time_year:                  float = 0.25 # change in time in years TODO: Update with game state's year level
  current_sea_level_mm:                 float = 0.0 # current sea level in mm TODO: Update with game state's sea level

  # # print(calculate_future_values(current_CO2_ppm, current_temperature_celcius, change_in_time_year))
  # test_temperatures = [10.0, 15.0, 20.0, 25.0, 30.0]
  # test_sea_levels = [0.0, 5.0, 10.0, 15.0, 20.0]

  # # Test 1: How different temperatures affect sea level rise
  # print("TEMPERATURE CHANGE TEST")
  # print("Starting Temp (°C) | Future Temp (°C) | Temp Change (°C)")
  # print("-" * 50)
  
  # for temp in test_temperatures:
  #     result = calculate_future_values(current_CO2_ppm, temp, current_sea_level, change_in_time_year)
  #     future_temp = result['future_temperature_celcius']
  #     temp_change = result['temperature_change']
      
  #     print(f"{temp:13.1f} | {future_temp:12.1f} | {temp_change:11.1f}")
  
  # # Test 2: How different starting sea levels progress (using varying temperatures)
  # print(f"\nSEA LEVEL TEST AT DIFFERENT TEMPERATURES")
  # print("Starting Sea Level (mm) | Starting Temp (°C) | Future Sea Level (mm) | Sea Level Change (mm)")
  # print("-" * 95)
  
  # for i, sea_level in enumerate(test_sea_levels):
  #     # Use different temperatures for each test to show variation
  #     temp_for_test = test_temperatures[i] if i < len(test_temperatures) else 20.0
  #     result = calculate_future_values(current_CO2_ppm, temp_for_test, sea_level, change_in_time_year)
  #     future_sea_level = result['future_sea_level_mm']
  #     sea_level_change = result['sea_level_change']
      
  #     print(f"{sea_level:20.1f} | {temp_for_test:15.1f} | {future_sea_level:17.1f} | {sea_level_change:17.1f}")
  future_temperature, sea_level_rise = calculate_future_values(current_CO2_ppm, current_temperature_celcius, current_sea_level_mm, change_in_time_year)
  
  print(f"Temperature: {future_temperature:.2f}°C, Sea Level Rise: {sea_level_rise:.2f} mm")

if __name__ == "__main__":
  main()
