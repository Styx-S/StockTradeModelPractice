import pandas as pd
import numpy as np

def tradeStrategy(init_fund, data):
    data_length = data.shape[0]
    start_idx = 10
    cmds = []
    volumn = 0
    fund = init_fund
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
                sall_vol = volumn * (trend / 10)
                fund += sall_vol * data.ix[i]['average']
                cmds.append((index, op, sall_vol))
                volumn -= sall_vol
        elif trend < 0:
            op = 0
            index = i
            buy_cost = (-trend / 10) * 0.5 * fund
            buy_vol = buy_cost // data.ix[i]['average']
            buy_cost = buy_vol * data.ix[i]['average']
            fund -= buy_cost
            cmds.append((index, op, buy_vol))
            volumn += buy_vol

    return cmds, fund, volumn


