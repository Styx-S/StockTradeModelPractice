import numpy as np
import math


# calculate the cosine similarity of two vector
def ccs(a, b):
    if len(a) != len(b):
        raise Exception("Invalid arguments", (a, b))
    a = np.mat(a)
    b = np.mat(b)
    num = float(a * b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = num / denom
    sim = 0.5 + cos * 0.5
    return sim


def sigmoid(a):
    if a > 0:
        return 1
    elif a < 0:
        return 0
    else:
        return a


def sliding_predict_model(initial_fund, test_set, train_set, length = 10, step = 0.2, price_indexes ="average", cmd=(0, 1)):
    fund = initial_fund
    had = 0
    price_list = list(train_set[price_indexes])
    train_data = list()
    # learn the underlying rule (if it does exist)
    for i in range(len(price_list)):
        if i > len(price_list) - length -1:
            break
        seq = list()
        seq.append(price_list[i])
        for j in range(1, length - 1):
            seq.append(price_list[i+j] / seq[0])
        # save the seq and the price at last
        train_data.append((seq, price_list[i + length -1] - seq[0]))
    # collect the latest (length -1) days' prices and predict the next day's price
    predict_list = list()
    cmds = list()
    for index, day in test_set.iterrows():
        predict_list.append(day[price_indexes])
        if len(predict_list) < length - 1:
            continue
        elif len(predict_list) == length - 1:
            sum = 0
            for vector, weight in train_data:
                # calculate the similarity and whether it's profit or loss
                sum += ccs(predict_list, vector) * weight
            # the result -- weight sum
            sum /= len(train_data)
            # if sum > 0 buy some stocks (means will strengthen), else sell some stocks
            if sum > 0:
                nums2buy = math.floor(sigmoid(sum) * step * fund / day[price_indexes])
                if nums2buy <= 0:
                    continue
                had += nums2buy
                fund -= nums2buy * day[price_indexes]
                cmds.append((index, cmd[0], nums2buy))
            else:
                nums2sell = math.floor(sigmoid(-sum) * step * had)
                if nums2sell <= 0:
                    continue
                had -= nums2sell
                fund += nums2sell * day[price_indexes]
                cmds.append((index, cmd[1], nums2sell))
            predict_list.pop(0)
    return cmds, fund, had


if __name__ == "__main__":
    a = np.array([1, 0.9, 0.7])
    b = np.array([1, 0, 0])
    print(ccs(a, b))
