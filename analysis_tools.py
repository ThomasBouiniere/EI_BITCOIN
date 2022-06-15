import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("data/df_blockchain.csv")["market-price"]


def mma(df, n):
    return pd.DataFrame(
        [np.mean(df[i - n: i]) for i in range(n, len(df))], columns=["MA{}".format(n)]
    )


def mme(df, m, approx=0):
    # 0 < m < 1
    if approx == 0:
        moy, mn = [df[0]], df[0]
        for i in range(1, len(df)):
            mn = mn + m * (df[i] - mn)
            moy.append(mn)
        return pd.DataFrame(moy, columns=["MME{}".format(m)])
    else:
        exp = np.flip(np.cumprod(m * np.ones(approx)))
        return pd.DataFrame(
            [
                np.dot(exp, np.array(df[i - approx: i])) / np.sum(exp)
                for i in range(approx, len(df))
            ],
            columns=["MME{}_{}".format(m, approx)],
        )


def macd(df, m, t1, t2, period=9):
    # 0 < m < 1, t1 < t2
    res = pd.DataFrame(
        np.array(mme(df, m, t1)[t2 - t1:]) - np.array(mme(df, m, t2)),
        columns=["MACD{}_{}".format(t1, t2)],
    )
    signal = mme(res, m, period)
    signal.columns = ["SGNL"]
    macdh = pd.DataFrame(
        np.array(res[period:]) - np.array(signal), columns=["MACDH"])
    return res, signal, macdh


def bollinger(df, m, n=12):
    d = {
        "BOLL{}_{}+".format(m, n): [
            np.mean(df[i - m: i]) + n * np.std(df[i - m: i])
            for i in range(m, len(df))
        ],
        "BOLL{}_{}-".format(m, n): [
            np.mean(df[i - m: i]) - n * np.std(df[i - m: i])
            for i in range(m, len(df))
        ],
        "BOLLWDTH{}_{}".format(m, n): [
            2 * n * np.std(df[i - m: i]) for i in range(m, len(df))
        ],
    }
    return pd.DataFrame(data=d)


def deltamma(df, n1, n2):
    #n2 > n1
    return pd.DataFrame(
        np.array(mma(df, n1)[n2 - n1:]) - np.array(mma(df, n2)),
        columns=["MA{}_{}".format(n1, n2)],
    )
