import pandas as pd
import numpy as np
import datetime

datadir='G:/pythonWorkspace/Huangbaoche/data/'
feature_url='G:/pythonWorkspace/Huangbaoche/feature/'

def read_data():

    orderHistory_tr = pd.read_csv(datadir + 'trainingset/orderHistory_train.csv')
    orderHistory_ts = pd.read_csv(datadir + 'test/orderHistory_test.csv')

    # 转换时间
    # datetime.datetime.utcfromtimestamp(timestamp) # 时区为UTC
    orderHistory_tr['orderTime_decode'] = orderHistory_tr.orderTime.apply(datetime.datetime.utcfromtimestamp)
    orderHistory_ts['orderTime_decode'] = orderHistory_ts.orderTime.apply(datetime.datetime.utcfromtimestamp)

    orderHistory_tr['orderTime_decode_year'] = orderHistory_tr['orderTime_decode'].apply(lambda x: x.year)
    orderHistory_tr['orderTime_decode_month'] = orderHistory_tr['orderTime_decode'].apply(lambda x: x.month)
    orderHistory_tr['orderTime_decode_day'] = orderHistory_tr['orderTime_decode'].apply(lambda x: x.day)

    orderHistory_ts['orderTime_decode_year'] = orderHistory_ts['orderTime_decode'].apply(lambda x: x.year)
    orderHistory_ts['orderTime_decode_month'] = orderHistory_ts['orderTime_decode'].apply(lambda x: x.month)
    orderHistory_ts['orderTime_decode_day'] = orderHistory_ts['orderTime_decode'].apply(lambda x: x.day)

    # 转换city
    mapdict = {}
    df = orderHistory_tr
    s = orderHistory_tr.city
    k = 20
    topk = s.value_counts()[:k].index.tolist()
    for k in topk:
        mapdict[k] = k
    for k in s.value_counts()[k:].index.tolist():
        mapdict[k] = '其他'
    # print(mapdict)
    orderHistory_tr['city_coded'] = coding(orderHistory_tr.city, mapdict)
    orderHistory_ts['city_coded'] = coding(orderHistory_ts.city, mapdict)

    # 转换国家
    mapdict = {}
    df = orderHistory_tr
    s = orderHistory_tr.country
    k = 20
    topk = s.value_counts()[:k].index.tolist()
    for k in topk:
        mapdict[k] = k
    for k in s.value_counts()[k:].index.tolist():
        mapdict[k] = '其他'
    # print(mapdict)
    orderHistory_tr['country_coded'] = coding(orderHistory_tr.country, mapdict)
    orderHistory_ts['country_coded'] = coding(orderHistory_ts.country, mapdict)


    return orderHistory_tr,orderHistory_ts



def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

def build_feature(orderHistory):
    orderHistory_fea = pd.get_dummies(orderHistory, columns=['city_coded', 'country_coded', 'continent'],
                                         prefix=['city_coded', 'country_coded', 'continent'])
    orderHistory_fea.drop(['city', 'country'], inplace=True, axis=1)
    # 每个用户历史订单总数
    orderHistory_feature = orderHistory_fea.groupby(['userid']).orderid.size().reset_index().rename(
        columns={'orderid': 'order_sum'})

    # 每个用户历史类型1订单总数
    df = orderHistory_fea[orderHistory_fea.orderType == 1].groupby(
        ['userid']).orderid.size().reset_index().rename(columns={'orderid': 'order_sum_type_1'})
    orderHistory_feature = pd.merge(orderHistory_feature, df, on=['userid'], how='left')

    # 每个用户历史类型0订单总数
    df = orderHistory_fea[orderHistory_fea.orderType == 0].groupby(
        ['userid']).orderid.size().reset_index().rename(columns={'orderid': 'order_sum_type_0'})
    orderHistory_feature = pd.merge(orderHistory_feature, df, on=['userid'], how='left')

    #每个用户每个旅游城市/国家/大陆的订单总数
    flist = orderHistory_fea.columns[8:].tolist()
    for fea in flist:
        df = orderHistory_fea.groupby('userid')[fea].sum().reset_index().rename(columns={fea: fea + '_sum'})
        orderHistory_feature = pd.merge(orderHistory_feature, df, on='userid', how='left')


    orderHistory_feature = orderHistory_feature.fillna(0)


    return orderHistory_feature


def writeToFile(feature):
    feature.to_csv(feature_url + 'orderHistory_ts_feature_v11.csv', sep=',', index=False, encoding='utf-8')

def main():
    orderHistory_tr, orderHistory_ts=read_data()
    orderHistory_feature=build_feature(orderHistory_ts)
    writeToFile(orderHistory_feature)

if __name__ == '__main__':
    main()