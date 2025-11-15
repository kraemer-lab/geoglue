"""CF-compliant attributes for common climatic variables"""

air_temperature = {
    "long_name": "2 meters air temperature",
    "units": "K",
    "standard_name": "air_temperature",
    "valid_min": 175,  # -98.15 °C, this is slightly less than the actual lowest
    "valid_max": 335,  # 61.85 °C, and highest air temperatures recorded
    "short_name": "t2m",
    "short_name_min": "mn2t",
    "short_name_max": "mx2t",
}

total_precipitation = {
    "long_name": "Total precipitation",
    "units": "m",
    "valid_min": 0,
    "short_name": "tp",
}

evaporation = {
    "long_name": "Evaporation",
    "units": "m",
    "short_name": "e",
}

relative_humidity = {
    "long_name": "Relative humidity",
    "depends": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure"],
    "valid_min": 0,
    "valid_max": 1,
    "standard_name": "relative_humidity",
    "units": "1",
    "short_name": "rh",
}

relative_humidity_percent = {
    "long_name": "Relative humidity",
    "depends": ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure"],
    "valid_min": 0,
    "valid_max": 100,
    "standard_name": "relative_humidity",
    "units": "percent",
    "short_name": "rh",
}

specific_humidity = {
    "long_name": "Specific humidity",
    "valid_min": 0,
    "standard_name": "specific_humidity",
    "units": "g kg-1",
    "short_name": "q",
}

hydrological_balance = {
    "long_name": "Hydrological balance",
    "valid_min": 0,
    "units": "m",
    "short_name": "hb",
}

spi = {
    "long_name": "Standardised precipitation",
    "units": "1",
    "short_name": "spi",
}

spei = {
    "long_name": "Standardised precipitation-evaporation index",
    "units": "1",
    "short_name": "spei",
}


def bias_corrected(d: dict) -> dict:
    out = d.copy()
    out["long_name"] = out["long_name"] + " (bias_corrected)"
    out["short_name"] = out["short_name"] + "_bc"
    return out
