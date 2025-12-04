#There is no universal formula for calculating future sea level rise so we are using a
#makeshift model based on the current implementation of the game mechanics thus far.
def calculate_new_sea_level(current_sea_level_mm: float, current_temperature_celcius: float, change_in_time_years: float) -> float:
    """
    Calculate the future value of sea level (mm) from the current sea level (mm) and global average temperature (celcius), as well as the amount of time elapsed in years.
    """
    # Baseline pre-industrial temperature
    baseline_temperature_celcius = 14.0  # °C
    
    # Rate of sea level rise per degree of warming per year
    rate_of_rise_per_degree_per_year = 3.3  # mm/°C/year
    
    # Calculate temperature increase from baseline
    temperature_increase = max(0, current_temperature_celcius - baseline_temperature_celcius)
    
    # Calculate the change in sea level based on warming only
    change_in_sea_level_mm = rate_of_rise_per_degree_per_year * temperature_increase * change_in_time_years
    
    # Find the new sea level
    future_sea_level_mm = current_sea_level_mm + change_in_sea_level_mm
    
    return future_sea_level_mm
