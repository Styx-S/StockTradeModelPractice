import pandas as pd
import numpy as np

def tradeStrategy(init_fund, data):
    data_length = data.shape[0]
    start_idx = 10
    cmds = []
    volumn = 0
    fund = init_fund
    can_buy = 0
    prev_fund = init_fund
    prev_assets = init_fund
    for i in range(start_idx, data_length):
        probe_idx = i - 10

        trend = 0
        for j in range(probe_idx, i):
            if data.ix[j]['average'] > data.ix[j+1]['average']:
                trend -= 1
            elif data.ix[j]['average'] < data.ix[j+1]['average']:
                trend += 1

        if trend > 0:
            if volumn > 0:
                op = 1
                index = i
                sall_vol = int(volumn * (trend / 10))
                if sall_vol == 0:
                    continue
                if fund + sall_vol * data.ix[i]['average'] + (volumn - sall_vol) * data.ix[i]['average'] - prev_assets > (0-prev_assets) * 0.01:
                    fund += sall_vol * data.ix[i]['average']
                    cmds.append((index, op, sall_vol))
                    volumn -= sall_vol

                    if fund > prev_fund:
                        prev_fund = fund

                    if can_buy == 1:
                        if fund >= 0.8 * prev_fund:
                            can_buy = 2
                    prev_assets = fund + sall_vol * data.ix[i]['average'] + volumn * data.ix[i]['average']


        elif trend < 0:
            op = 0
            index = i
            if (can_buy == 2 or can_buy == 0):
                buy_cost = (-trend / 10) * 0.5 * fund
                buy_vol = buy_cost // data.ix[i]['average']
                buy_cost = buy_vol * data.ix[i]['average']

                if fund - buy_cost + (volumn + buy_vol) * data.ix[i]['average'] > (0-prev_assets) * 0.01:
                    fund -= buy_cost
                    cmds.append((index, op, buy_vol))
                    volumn += buy_vol

                    if fund <= prev_fund * 0.5:
                        can_buy = 1

                    prev_assets = fund - buy_cost + volumn * data.ix[i]['average'] >= prev_assets

    return cmds, fund, volumn


