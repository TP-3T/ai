# columns from Interim_Climate_Dataset_from_1940_baseline_1950_FINAL.csv

# === Semantic and ordering ===
ROW_ID: str = "ROW_ID"
YEAR: str = "Year"

# === FEATURES ===

CO2_PPM: str = "CO2 (ppm)"
TEMP_DEG_C: str = "TEMP (deg C)"
ABSOLUTE_GMSL_MM_RELATIVE_TO_JAN_1950: str = "Absolute GMSL (mm) relative to Jan 1950"

CO2_3M_AGO: str = "CO2_3m_ago"
CO2_6M_AGO: str = "CO2_6m_ago"
CO2_12M_AGO: str = "CO2_12m_ago"
CO2_5Y_AGO: str = "CO2_5y_ago"
CO2_10Y_AGO: str = "CO2_10y_ago"

DELTA_CO2_FROM_3M_AGO: str = "Change_in_CO2_from_3m_ago"
DELTA_CO2_FROM_6M_AGO: str = "Change_in_CO2_from_6m_ago"
DELTA_CO2_FROM_12M_AGO: str = "Change_in_CO2_from_12m_ago"
DELTA_CO2_FROM_5Y_AGO: str = "Change_in_CO2_from_5y_ago"
DELTA_CO2_FROM_10Y_AGO: str = "Change_in_CO2_from_10y_ago"

TEMP_3M_AGO: str = "TEMP_3m_ago"
TEMP_6M_AGO: str = "TEMP_6m_ago"
TEMP_12M_AGO: str = "TEMP_12m_ago"
TEMP_5Y_AGO: str = "TEMP_5y_ago"
TEMP_10Y_AGO: str = "TEMP_10y_ago"

DELTA_TEMP_FROM_3M_AGO: str = "Change_in_TEMP_from_3m_ago"
DELTA_TEMP_FROM_6M_AGO: str = "Change_in_TEMP_from_6m_ago"
DELTA_TEMP_FROM_12M_AGO: str = "Change_in_TEMP_from_12m_ago"
DELTA_TEMP_FROM_5Y_AGO: str = "Change_in_TEMP_from_5y_ago"
DELTA_TEMP_FROM_10Y_AGO: str = "Change_in_TEMP_from_10y_ago"

GMSL_3M_AGO: str = "GMSL_3m_ago"
GMSL_6M_AGO: str = "GMSL_6m_ago"
GMSL_12M_AGO: str = "GMSL_12m_ago"
GMSL_5Y_AGO: str = "GMSL_5y_ago"
GMSL_10Y_AGO: str = "GMSL_10y_ago"

DELTA_GMSL_FROM_3M_AGO: str = "Change_in_GMSL_from_3m_ago"
DELTA_GMSL_FROM_6M_AGO: str = "Change_in_GMSL_from_6m_ago"
DELTA_GMSL_FROM_12M_AGO: str = "Change_in_GMSL_from_12m_ago"
DELTA_GMSL_FROM_5Y_AGO: str = "Change_in_GMSL_from_5y_ago"
DELTA_GMSL_FROM_10Y_AGO: str = "Change_in_GMSL_from_10y_ago"

# === TARGETS ===

FUTURE_TEMP_TARGET_COL_NAME: str = "TEMP_after_3m_TO_PREDICT"
FUTURE_GMSL_TARGET_COL_NAME: str = "GMSL_after_3m_TO_PREDICT"
