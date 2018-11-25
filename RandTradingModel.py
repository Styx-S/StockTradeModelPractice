import pandas as pd
import math
import random


"""
@return cmds shows the suggestions this model gives,
        fund and had showed the status in the last day
"""


def rand_trading_model(initial_fund, test_set, train_set = None,
                       price_indexes ="average", MIN= 0.1, MAX= 0.3, cmd=(0, 1)):
    cmds = list()
    had = 0
    fund = initial_fund
    bool_dice = (True, False)
    for index, day in test_set.iterrows():
        weight = random.uniform(MIN,MAX)
        today_price = day[price_indexes]
        if random.choice(bool_dice):
            # buy some stocks
            nums2buy = math.floor(fund / today_price * weight)
            if nums2buy <= 0:
                continue
            else:
                had += nums2buy
                fund -= nums2buy * today_price
                cmds.append((index, cmd[0], nums2buy))
        else:
            # sell some stocks
            nums2sell = math.floor(had * weight)
            if nums2sell <= 0:
                continue
            else:
                had -= nums2sell
                fund += nums2sell * today_price
                cmds.append((index, cmd[1], nums2sell))
    return cmds, fund, had


if __name__ == "__main__":
    pass
