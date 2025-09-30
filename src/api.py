import requests
from typing import Union

import pandas as pd


def get_meter_data(ids: Union[str, list[str]], start_datetime: str, end_datetime: str, api_key: str, tz: str = "Europe/Rome",
                   aggregation: str = "15 m", data_format: str = "raw", tag: str = "ok") -> Union[dict, str]:
    """
    Fetches meter data from the CrownLabs BAEDA API.

    Parameters:
        ids (str): Meter ID(s), comma-separated if multiple.
        tz (str): Timezone, e.g. 'Europe/Rome'.
        start_datetime (str): Start datetime in 'YYYY-MM-DD HH:MM:SS' format.
        end_datetime (str): End datetime in 'YYYY-MM-DD HH:MM:SS' format.
        api_key (str): API key for authentication.
        aggregation (str): Aggregation level, default '15 m'.
        data_format (str): Data format, default 'raw', but 'csv' and 'txt' implemented.
        tag (str): Tag filter, default 'ok'.

    Returns:
        dict or str: Parsed JSON data or raw text on error.
    """
    url = "https://api.baeda.crownlabs.polito.it/data/meters"
    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }
    params = {
        "ids": ids,
        "tz": tz,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "aggregation": aggregation,
        "data_format": data_format,
        "tag": tag
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raises HTTPError if the response was unsuccessful
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return response.text if 'response' in locals() else str(e)


def create_df(data: list) -> pd.DataFrame:
    """
    Converts the list of dictionaries (return of get_meter_data) to a pandas DataFrame.

    Parameters:
        data (list): List of dictionaries containing meter data.

    Returns:
        pd.DataFrame: DataFrame with the meter data.
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["time_15m"] = pd.to_datetime(df["time_15m"])
    # Convert meter_id to str
    df["meter_id"] = df["meter_id"].astype(str)
    pivot_df = df.pivot(index="time_15m", columns="meter_id", values="avg_value")
    pivot_df = pivot_df.sort_index()
    # Reset index
    pivot_df = pivot_df.reset_index(names="timestamp")

    # Convert timestamp to datetime without timezone
    pivot_df["timestamp"] = pd.to_datetime(pivot_df["timestamp"], utc=True).dt.tz_convert("Europe/Rome").dt.tz_localize(None)

    return pivot_df


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import pandas as pd
    import numpy as np
    load_dotenv()
    API_KEY_POLITO = os.getenv("API_KEY_POLITO")

    # Example usage
    data = get_meter_data(
        ids=["1283", "1284", "1305", "1306"],
        tz="Europe/Rome",
        start_datetime="2024-03-01 00:00:00",
        end_datetime="2025-05-31 23:59:00",
        api_key=API_KEY_POLITO
    )

    df = create_df(data)

    df["1307"] = df["1305"] + df["1306"]
    df["gross"] = np.nan
    df["baseload"] = np.nan
    # Between 16:30 and 7:00, if 1307 is nan, put 0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    mask = ((df["hour"] >= 16) | (df["hour"] < 7)) & (df["1307"].isna() & df["1305"].isna() & df["1306"].isna())
    df.loc[mask, "1307"] = 0

    # Se 1307 < 1284 -> gross = 1284 + 1307 e baseload = gross - 1284
    df.loc[df["1307"] < df["1284"], "gross"] = df["1307"] + df["1283"]
    df.loc[df["1307"] < df["1284"], "baseload"] = df["gross"] - df["1284"]

    # Group by weekend days (saturday and sunday) and weekdays (monday to friday), and calculate the average baseload
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Create a column for weekday, saturday, and sunday
    df["day_type"] = np.where(df["day_of_week"] < 5, "weekday",
                                np.where(df["day_of_week"] == 5, "saturday", "sunday"))
    df["baseload"] = df["baseload"].astype(float)
    df_grouped = df.groupby(["day_type", "hour", "minute"])["baseload"].mean().reset_index()

    # When baseload is NaN, fill with the corresponding average value un df_grouped, in the same day_type,hour and minute
    for _, row in df_grouped.iterrows():
        mask = (df["day_type"] == row["day_type"]) & (df["hour"] == row["hour"]) & (df["minute"] == row["minute"])
        df.loc[mask & df["baseload"].isna(), "baseload"] = row["baseload"]

    # When gross is NaN, perform  baseload + 1284
    df["gross"] = df["gross"].fillna(df["baseload"] + df["1284"])

    df["Cabina F1"] = df["gross"]
    df["Aule P"] = df["Cabina F1"]
    df["Gen QE-CDZ"] = df["1284"]
    df["Unlabelled"] = df["baseload"]

    df = df[["timestamp", "Cabina F1", "Aule P", "Gen QE-CDZ", "Unlabelled"]]

    for column in df.columns:
        if column != "timestamp":
            df_save = df[["timestamp", column]].rename(columns={column: "value"})
            df_save.to_csv(f"{column}.csv", index=False)




