# Energy balance model
# ASR (absorbed shortwave radiation - from the sun)
# OLR (outgoing longwave radiation - from Earth)

# Heat gets trapped in atmosphere when it decreases ability to allow radiation through (caused by increase in CO2 and other atmospheric gases)
# as a result OLR decreases

# Energy balance of Earth is achieved when ASR = OLR, global warming occurs when OLR has to increase to be equal to ASR

# === Outgoing Earth Radiation ===

# Equation for OLR: OLR = tau * sigma * T^4

# define the Stefan-Boltzmann Constant, 'e' for scientific notation
sigma = 5.67e-8  # W m^-2 K^-4

# # define the outgoing longwave radiation based on observations from the IPCC AR6 Figure 7.2
# OLR = 239  # W m^-2

# # define the emission temperature based on observations of global mean surface temperature
# T = 288  # K

# transmissivity of the atmosphere (changes with concentration of greenhouse gases)
# tau = OLR / (sigma * T**4)  # unitless number between 0 and 1

# === tau will stay constant for our simulation, but the CO2 radiative forcing value will contribute to a change in the final temperature

tau = 0.6127 # for our simulation, will make it a constant value (derived from IPCC AR6 OLR value)

# === Incoming Solar Radiation ===

# insolation - insolation, the total amount of solar radiation received at the earth's surface - https://glossary.ametsoc.org/wiki/Insolation
# reflected_flux - the amount of radiation reflected back to space (reflected of ice/snow/clouds)
# albedo - planet albedo, the fraction of reflected radiation relative to the insolation - https://glossarytest.ametsoc.net/wiki/Planetary_albedo

# ASR - absorbed shortwave radiation - the amount of insolation NOT reflected (amount that gets to Earth surface)

# the observed insolation based on observations from the IPCC AR6 Figure 7.2
insolation = 340  # W m^-2

# the observed reflected radiation based on observations from the IPCC AR6 Figure 7.2
reflected_flux = 100  # W m^-2

# stays constant for our simulation
albedo = reflected_flux / insolation # unitless number between 0 and 1
# albedo = 0.2941

# ==========

# Calculate heat capacity of the Earth system

# heat capacity of the ocean
c_oc = 3850     # specific heat of seawater in J/kg/K
rho_oc = 1025   # average density of seawater in kg/m3
d_oc = 70       # depth of water in m (here representative of the mixed layer)
C_oc = c_oc * rho_oc * d_oc  # heat capacity of the ocean

# heat capacity of the atmosphere
c_atm = 1004    # specific heat of the atmosphere at constant pressure in J/kg/K
W_atm = 100000  # weight (pressure) of atmospheric column in Pa
g = 9.81        # acceleration due to gravity in m/s^2
C_atm = c_atm * (W_atm / g)  # heat capacity of the atmosphere

# total heat capacity of the earth system
C_EARTH = C_oc + C_atm

# ===

__PREINDUSTRIAL_TEMPERATURE_KELVIN: float = 288.0

# Net climate feedback parameter 
# The amount of response to a change in global surface temperature (W/m²/K)
# (to account for water vapour, clouds, and the planet emitting heat)
# This is an approximation
__LAMBDA_W_PER_M2_PER_K: float = 1.1

def albedo_from_temperature(current_temperature_kelvin: float) -> float:
    """
    Simple ice albedo feedback (adapted from Climatematch).
    """
    if current_temperature_kelvin >= 300.0:
        # no albedo (warm with no ice)
        return 0.1      
    elif current_temperature_kelvin > 240.0:
        # between high albedo and low
        return 0.1 + (0.7 - 0.1) * (current_temperature_kelvin - 300.0) ** 2 / (240.0 - 300.0) ** 2
    else:
        # high albedo (cold)
        return 0.7


# absorbed shortwave radiation (ASR)
# Equation for ASR: ASR = insolation - reflected_flux = (1 - alpha) * insolation
def __ASR(current_temperature_kelvin: float):
  """
  Calculate the absorbed shortwave radiation - the amount of solar energy absorbed by Earth.
  """
  albedo: float = albedo_from_temperature(current_temperature_kelvin)
  return (1 - albedo) * insolation

__ASR_PREINDUSTRIAL: float = __ASR(__PREINDUSTRIAL_TEMPERATURE_KELVIN)

# # outgoing longwave radiation (OLR)
# def __OLR(current_temperature: float) -> float:
#   return tau * sigma * current_temperature**4

def calculate_net_radiative_imbalance(
  current_temperature_kelvin: float,
  co2_radiative_forcing_watts_per_metre_sq: float      
):
  """
  Calculate the difference between the amount of solar energy absorbed 
  by Earth and the amount of energy the planet radiates back into space (in W/m^2).
  net_readiative_imbalance = [ASR(T) - ASR_pre_industrial] + F_CO2 - λ (T_current - T_pre_industrial) 
  """
  # change in ASR from the change in the amount of albedo (the amount of radiation reflected by Earth ice, clouds etc.)
  change_in_ASR: float = __ASR(current_temperature_kelvin) - __ASR_PREINDUSTRIAL

  change_in_temp_from_preindustrial: float = current_temperature_kelvin - __PREINDUSTRIAL_TEMPERATURE_KELVIN

  net_radiative_imbalance: float = change_in_ASR \
  + co2_radiative_forcing_watts_per_metre_sq \
  - (__LAMBDA_W_PER_M2_PER_K * change_in_temp_from_preindustrial)

  return net_radiative_imbalance

# finds the new temperature based on the previous using Forward Euler method (approximates derivitive to change in value)
def calculate_new_temperature(current_temperature_kelvin: float, change_in_time_seconds: float, co2_radiative_forcing_watts_per_metre_sq: float) -> float:
  """
  Calculate the future value of temperature (Kelvin) from the current global average temperature (celcius) and CO2 radiative forcing (Watts / metre^2), as well as the amount of time elapsed in years.
  """
  net_radiative_imbalance: float = calculate_net_radiative_imbalance(current_temperature_kelvin, co2_radiative_forcing_watts_per_metre_sq)

  # change_in_temperature: float = (change_in_time_seconds / C_EARTH) * ( __ASR(current_temperature_kelvin) + co2_radiative_forcing_watts_per_metre_sq - __OLR(current_temperature_kelvin) )

  change_in_temperature: float = (change_in_time_seconds / C_EARTH) * net_radiative_imbalance

  # find the new temperature using forward Euler method
  future_temperature: float = current_temperature_kelvin + change_in_temperature

  return future_temperature

# if __name__ == "__main__":
#   # Equilibrium temperature test
#   current_temperature_celcius: float = 50.0 # current temperature in Celsius TODO: Update with game state's current temperature
#   current_temperature_kelvin:  float = current_temperature_celcius + 273.15
#   T_eq: float = ((__ASR(current_temperature_kelvin)) / (tau * sigma)) ** (1 / 4)
#   print("Equilibrium Temperature: ", T_eq, "K or", T_eq - 273, "°C")
