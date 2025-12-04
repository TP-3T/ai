import math
# Empirical constant for CO2 forcing, derived from radiative transfer calculations that is specific to CO2
__EMPIRICAL_CONSTANT: float = 5.35

# Pre-industrial CO2 concentration in ppm
__CO2_PRE_INDUSTRIAL:  float = 280.0

def calculate_radiative_forcing(current_co2_concentration_ppm: float):
  """Calculate the radiative forcing value (Watts / metre^2) based on the current CO2 concentration."""

  # radiative forcing: the amount of change in Earth's energy balance due to CO2 (W / m^2)
  # if positive value, global warming is occuring

  co2_radiative_forcing: float = __EMPIRICAL_CONSTANT * math.log(current_co2_concentration_ppm / __CO2_PRE_INDUSTRIAL)
  return co2_radiative_forcing
