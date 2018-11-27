import tushare as ts
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import datetime
import mpl_finance as mpf
from RandTradingModel import rand_trading_model
from trad_strategy import tradeStrategy
from SlidingPredictModel import sliding_predict_model

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

# tech： 科大讯飞002230 海康威视002415 浪潮信息000977 华大基因300676 360安全601360
# enc： 平安银行000001 国泰君安601211 光大证券601788 中国人寿601628 中国银行601988
# metal：太钢不锈000825 马钢股份600808 宝钢股份600019 鞍钢股份000898 河钢股份000709
# power: 西昌电力600505 岷江水电600131 东旭蓝天000040 惠天热电000692 东方能源000958
# shipping：唐山港601000 营口港600317 宁波海运600798 连云港601008 大连港601880 天津港600717
# housing： 亚太实业000691 ST宏盛600817 中国武夷000797 广宇集团002133 信达地产600657
# tourism： 西安旅游000610 云南旅游002059 大连圣亚600593 中国国旅601888 ST藏旅600749


STOCK_LIST = [TECH_SOTCK, ENC_SOTCK, METAL_STOCK, POWER_STOCK, SHIPPING_STOCK, HOUSING_STOCK, TOURISM_STOCK]

START_TIME = '2010-01-01'
END_TIME = '2018-11-01'

OP_BUY = 0
OP_SALL = 1

NONTRAINABLE = False
TRAINABLE = True
DATA_ROOT_PATH = './data/'

def fetchData(id, start_time, end_time):
    data = ts.get_hist_data(id, start=start_time, end=end_time, ktype='D')
    return data

def loadData(id):
    names = os.listdir('./data/')
    filename = ''
    for name in names:
        if re.match(id, name):
            filename = name
            break
    if filename != '':
        data = pd.read_csv(f'./data/{filename}')
        data['average'] = (data['high'] + data['low'])/2
        data = data.iloc[::-1]
        data = data.reset_index(drop=True)
        return data
    else:
        return None

def trainTestSplit(data, train_ratio=0.8):
    data_length = data.shape[0]
    train_count = int(data_length * train_ratio)
    train_set = data.iloc[:train_count]
    test_set = data.iloc[train_count:]
    return train_set, test_set

def runModel(model, data, init_fund, trainable=False):
    train_set, test_set = trainTestSplit(data)
    if trainable == True:
        cmds, fund, had = model(init_fund, test_set, train_set)

    else:
        cmds, fund, had = model(init_fund, data)
        last_op = cmds[-1]
        last_price = data.ix[last_op[0]]['average']
        current_stock_value = had * last_price
        operation_count = len(cmds)

        print(fund)
        print(fund + current_stock_value)

    current_stock_count = 0
    fund = init_fund
    funds = [fund]
    total_assets = [fund]
    stock_list = [0]

    for index, op, stocks in cmds:
        if op == OP_BUY:
            current_stock_count += stocks
            stock_list.append(current_stock_count)
            buy_stock_values = stocks * data.ix[index]['average']
            fund = fund - buy_stock_values
            funds.append(fund)
            total_assets.append(fund + current_stock_count * data.ix[index]['average'])
        elif op == OP_SALL:
            current_stock_count -= stocks
            stock_list.append(current_stock_count)
            sall_stock_values = stocks * data.ix[index]['average']
            fund = fund + sall_stock_values
            funds.append(fund)
            total_assets.append(fund + current_stock_count * data.ix[index]['average'])

    plt.plot(funds, 'o-')
    plt.plot(total_assets)
    plt.grid()
    plt.show()

    return funds, total_assets, stock_list




def drawAverage(data):
    T = []
    ave = []
    for date, row in data.iterrows():
        date_time = datetime.datetime.strptime(row['date'], "%Y-%m-%d")
        t = date2num(date_time)
        average = row['average']
        T.append(t)
        ave.append(average)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis_date()
    plt.xticks(rotation=45)
    plt.yticks()
    plt.xlabel('time')
    plt.ylabel('average price')
    plt.plot(T, ave)
    plt.grid()
    plt.show()

def drawKLineDiagram(data):
    data_list = []
    for date, row in data.iterrows():
        date_time = datetime.datetime.strptime(row['date'], "%Y-%m-%d")
        t = date2num(date_time)
        open, high, close, low = row[1:5]
        datas = (t, open, high, low, close)
        data_list.append(datas)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis_date()
    plt.xticks(rotation=45)
    plt.yticks()
    plt.xlabel('time')
    plt.ylabel('price')
    mpf.candlestick_ohlc(ax, data_list, width=1.5, colorup='r', colordown='green')
    plt.grid()
    plt.show()


def pipeline(model, init_fund, model_type=NONTRAINABLE):

    if not os.path.exists(model.__name__):
        os.mkdir(model.__name__)
    model_profile_path = './' + model.__name__

    file_name = os.listdir(DATA_ROOT_PATH)
    for name in file_name:
        if not os.path.exists(model_profile_path + '/' + os.path.splitext(name)[0]):
            os.mkdir(model_profile_path + '/' + os.path.splitext(name)[0])

        profile_path = model_profile_path + '/' + os.path.splitext(name)[0]
        data = loadData(name)
        funds, total_assets, stock_count = runModel(model, data, init_fund, model_type)
        record = pd.DataFrame({'funds': funds, 'total_assets': total_assets, 'stock_count': stock_count})
        record.to_csv(profile_path + '/' + 'profile.csv')




if __name__ == '__main__':
    # for stock_class in STOCK_LIST:
    #     print(getattr(stock_class, '__name__'))
    #     for k in stock_class.keys():
    #         file_name = f"./data/{k}_{stock_class[k]['name']}.csv"
    #         stock_class[k]['data'] = fetchData(k, START_TIME, END_TIME)
    #         try:
    #             stock_class[k]['data'].to_csv(file_name)
    #         except:
    #             print(k, stock_class[k]['name'])

    data = loadData('002230')
    runModel(sliding_predict_model, data, 10000, True)
    # drawKLineDiagram(data)
    # runModel(rand_trading_model, data, 10000, False)
    # pipeline(rand_trading_model, 10000, NONTRAINABLE)