#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 13/8/2024
import logging
import math
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import holidays
from jinja2 import Environment, FileSystemLoader

# Path to folders
path_to_data = os.path.join(os.path.dirname(__file__), 'data')
path_to_figures = os.path.join(os.path.dirname(__file__), 'results', 'figures')
path_to_reports = os.path.join(os.path.dirname(__file__), 'results', 'reports')
path_to_templates = os.path.join(os.path.dirname(__file__), 'templates')

color_palette = 'viridis'
dpi_resolution = 300
fontsize = 10
line_style_context = '-'
line_style_other = ':'
line_color_context = '#D83C3B'
line_color_other = '#D5D5E0'
line_size = 1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s](%(name)s) %(message)s')


def extract_holidays(data: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Extract holidays from the dataset and the country code

    :param data: data already processed
    :param country_code: country code to extract holidays

    :return: holidays dataframe
    """

    holidays_country = holidays.country_holidays(country_code)
    data['date'] = data.index.date

    dates = data['date'].unique()

    df_holidays = pd.DataFrame(columns=['holiday'])
    for date in dates:
        if date in holidays_country:
            df_holidays.loc[date, "holiday"] = holidays_country.get(date)

    return df_holidays

def hour_to_dec(hour_str: str) -> float:
    """ Transforms float hours from HH:MM string format to float with decimal places

    :param hour_str: hour in format HH:MM
    :type hour_str: str

    :return hour_dec: hour in numerical format
    :rtype hour_dec: float

    :example:
    >>> hour_to_dec('02:00')
    2.0
    """

    (H, M) = hour_str.split(':')
    hour_dec = int(H) + int(M) / 60
    return hour_dec


def dec_to_hour(hour_dec: float) -> str:
    """ Transforms float hours with decimal places into HH:MM string format

    :param hour_dec: hour in numerical format
    :type hour_dec: float

    :return hour_str: hour in format HH:MM
    :rtype hour_str: str

    :example:
    >>> dec_to_hour(2.5)
    '02:30'
    """

    (H, M) = divmod(hour_dec * 60, 60)
    hour_str = "%02d:%02d" % (H, M)
    return hour_str


def dec_to_obs(hour_dec, obs_per_hour):
    """  transforms float hours with decimal places into HH:MM string format

    :param hour_dec: hour interval in numerical format
    :type hour_dec: float

    :param obs_per_hour: number of observations per hour
    :type obs_per_hour: int

    :return observations: number of observations
    :rtype observations: int

    :example:
    >>> # 6.30 -> H = 6, M = 30
    >>> #6[hours]*4[observations/hour] + 30[minutes]*1/15[observations/minutes] = 25 [observations]
    >>> dec_to_obs(6.30 , 4)
    25
    """

    (H, M) = divmod(hour_dec * 60, 60)
    observations = int(H * obs_per_hour + M / 15)
    return observations



