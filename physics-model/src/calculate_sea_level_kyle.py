# Pre-industrial baseline temperature (global average)
__TEMPERATURE_PREINDUSTRIAL: float = 14.0  # °C

# Empirical sensitivity coefficients derived from IPCC observations
# Units: mm/year per °C of warming

# Thermal expansion sensitivity (responds quickly, within decades)
__THERMAL_EXPANSION_SENSITIVITY: float = 1.5  # mm/year/°C

# Ice melt sensitivity (responds more slowly, with lag)
# Includes: glaciers, Greenland ice sheet, Antarctic ice sheet
__ICE_MELT_SENSITIVITY: float = 1.5  # mm/year/°C

# Combined total sensitivity (thermal + ice)
__TOTAL_SENSITIVITY: float = __THERMAL_EXPANSION_SENSITIVITY + __ICE_MELT_SENSITIVITY  # 3.0 mm/year/°C

def calculate_sea_level_rise(
    future_temperature_celcius: float,
    time_elapsed_years: float
) -> float:
    """
    Parameters:
    -----------
    current_temperature_celcius : float
        Current global average temperature in degrees Celsius
    time_elapsed_years : float
        Time period over which to calculate sea level rise (years)
    
    Returns:
    --------
    float : Sea level rise in millimeters over the time period
    """
    
    # Calculate temperature anomaly relative to pre-industrial baseline
    temperature_anomaly: float = future_temperature_celcius - __TEMPERATURE_PREINDUSTRIAL
    
    # No sea level rise if temperature is at or below pre-industrial levels
    if temperature_anomaly <= 0:
        return 0.0
    
    # Calculate base annual rate of sea level rise
    annual_rate: float = __TOTAL_SENSITIVITY * temperature_anomaly  # mm/year
    
    # Calculate total rise over the time period
    total_rise_mm: float = annual_rate * time_elapsed_years
    
    return total_rise_mm


