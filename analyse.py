import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

TECH_SOTCK = {'002230': {'name': 'iflytek', 'data': None},
              '002415': {'name': 'hikvision', 'data': None},
              '000977': {'name': 'inspur', 'data': None},
              '300676': {'name': 'genomics', 'data': None},
              '601360': {'name': '360', 'data': None}}

ENC_SOTCK = {'000001': {'name': 'pingan', 'data': None},
             '601211': {'name': 'gtja', 'data': None},
             '601788': {'name': 'ebscn', 'data': None},
             '601628': {'name': 'chinalife', 'data': None},
             '601988': {'name': 'boc', 'data': None}}

METAL_STOCK = {'000825': {'name': 'tisco', 'data': None},
               '600808': {'name': 'masteel', 'data': None},
               '600019': {'name': 'baosteel', 'data': None},
               '000898': {'name': 'ansteel', 'data': None},
               '000709': {'name': 'hbgtgf', 'data': None}}

POWER_STOCK = {'600505': {'name': 'tisco', 'data': None},
               '600131': {'name': 'masteel', 'data': None},
               '000040': {'name': 'baosteel', 'data': None},
               '000692': {'name': 'ansteel', 'data': None},
               '000958': {'name': 'hbgtgf', 'data': None}}

SHIPPING_STOCK = {'600505': {'name': 'scxcdl', 'data': None},
                  '600131': {'name': 'mjsdgs', 'data': None},
                  '000040': {'name': 'dongxulantian', 'data': None},
                  '000692': {'name': 'htrd', 'data': None},
                  '000958': {'name': 'dfpower', 'data': None}}

SHIPPING_STOCK = {'601000': {'name': 'jtport', 'data': None},
                  '600317': {'name': 'ykport', 'data': None},
                  '600798': {'name': 'nbmc', 'data': None},
                  '601008': {'name': 'cosfrelyg', 'data': None},
                  '601880': {'name': 'portdalian', 'data': None}}

HOUSING_STOCK = {'000691': {'name': 'ytsy', 'data': None},
                  '600817': {'name': 'sttech', 'data': None},
                  '000797': {'name': 'chinawuyi', 'data': None},
                  '002133': {'name': 'cosmosgroup', 'data': None},
                  '600657': {'name': 'cindare', 'data': None}}

TOURISM_STOCK = {'000610': {'name': 'ctsxian', 'data': None},
                '002059': {'name': 'toynly', 'data': None},
                '600593': {'name': 'sunasia', 'data': None},
                '601888': {'name': 'cits', 'data': None},
                '600749': {'name': '600749', 'data': None}}

def VMAP(data, time_delta):
    n = data.shape[0]
    total_sum = 0
    volume_sum = 0
    for i in range(n):
        high_price = data['high']
        low_price = data['low']
        price = (high_price + low_price) / 2
        volume = data['volume']
        total_sum += price * volume
        volume_sum += volume
    return total_sum / volume_sum

def TMAP(data, time_delta):
    n = data.shape[0]
    price_sum = 0
    for i in range(n):
        high_price = data['high']
        low_price = data['low']
        price = (high_price + low_price) / 2
        price_sum += price
    return price_sum / n

if __name__ == '__main__':

    file_name = f"./data/002230_{TECH_SOTCK['002230']['name']}.csv"

    data = pd.read_csv(file_name)
    data = data.iloc[::-1, :].copy()
    data['date'] = pd.to_datetime(data['date'])
    time_delta = data['date'][1] - data['date'][31]


    plt.style.use('seaborn')
    plt.plot(data['date'], data['volume'])
    plt.plot(data['date'], data['high'])
    plt.plot(data['date'], data['close'])
    plt.show()
