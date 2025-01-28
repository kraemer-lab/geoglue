"""
Collate module for API-based retrievals.

These require direct downloads to file.
"""
import logging

import cdsapi


def fetch_era5():
    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": "2022",
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+07:00",
        "frequency": "6_hourly"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()

if __name__ == "__main__":
    fetch_era5()
