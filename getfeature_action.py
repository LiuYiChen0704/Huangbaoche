import pandas as pd
import numpy as np
import datetime

datadir='G:/pythonWorkspace/Huangbaoche/data/'
feature_url='G:/pythonWorkspace/Huangbaoche/feature/'

def read_data():
    action_tr = pd.read_csv(datadir + 'trainingset/action_train.csv')
    action_ts = pd.read_csv(datadir + 'test/action_test.csv')
    #时间处理
    action_tr['actionTime_decode'] = action_tr.actionTime.apply(datetime.datetime.utcfromtimestamp)
    action_ts['actionTime_decode'] = action_ts.actionTime.apply(datetime.datetime.utcfromtimestamp)

    action_tr['actionTime_decode_year'] = action_tr['actionTime_decode'].apply(lambda x: x.year)
    action_tr['actionTime_decode_month'] = action_tr['actionTime_decode'].apply(lambda x: x.month)
    action_tr['actionTime_decode_day'] = action_tr['actionTime_decode'].apply(lambda x: x.day)

    action_ts['actionTime_decode_year'] = action_ts['actionTime_decode'].apply(lambda x: x.year)
    action_ts['actionTime_decode_month'] = action_ts['actionTime_decode'].apply(lambda x: x.month)
    action_ts['actionTime_decode_day'] = action_ts['actionTime_decode'].apply(lambda x: x.day)

    return action_tr,action_ts


def coding(col, codeDict):

  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

def build_feature(action_df):
    # action feature
    actionType_dict = {3: 2, 4: 2}
    action_df['actionType_coded'] = coding(action_df.actionType, actionType_dict)
    action_fea = pd.get_dummies(action_df, columns=['actionType_coded'], prefix=['actionType_coded'])

    # 每个用户action最早和最晚时间差了几天（单位是天）
    s = action_fea.groupby(['userid']).actionTime_decode.max() - action_fea.groupby(
        ['userid']).actionTime_decode.min()
    action_feature = pd.DataFrame({'max_min_interval': s.apply(lambda x: x.days)}).reset_index()


    # actionType_coded_1,2,5,6,7,8,9的mean(),std(),min(),max()----按天group
    flist = ['actionType_coded_1', 'actionType_coded_2', 'actionType_coded_5', 'actionType_coded_6',
             'actionType_coded_7', 'actionType_coded_8', 'actionType_coded_9']
    for f in flist:
        tdf = action_fea.groupby(['userid', 'actionTime_decode_year', 'actionTime_decode_month', 'actionTime_decode_day'])[
            f].sum().reset_index()
        df1 = tdf.groupby(['userid'])[f].mean().reset_index().rename(columns={f: f + '_day_mean'})
        df2 = tdf.groupby(['userid'])[f].std().reset_index().rename(columns={f: f + '_day_std'})
        df3 = tdf.groupby(['userid'])[f].min().reset_index().rename(columns={f: f + '_day_min'})
        df4 = tdf.groupby(['userid'])[f].max().reset_index().rename(columns={f: f + '_day_max'})

        action_feature = action_feature.merge(df1, on=['userid'], how='left')
        action_feature = action_feature.merge(df2, on=['userid'], how='left')
        action_feature = action_feature.merge(df3, on=['userid'], how='left')
        action_feature = action_feature.merge(df4, on=['userid'], how='left')

    # actionType_coded_1,2,5,6,7,8,9的mean(),std(),min(),max()----按月group
    for f in flist:
        tdf = action_fea.groupby(['userid', 'actionTime_decode_year', 'actionTime_decode_month'])[
            f].sum().reset_index()
        df1 = tdf.groupby(['userid'])[f].mean().reset_index().rename(columns={f: f + '_month_mean'})
        df2 = tdf.groupby(['userid'])[f].std().reset_index().rename(columns={f: f + '_month_std'})
        df3 = tdf.groupby(['userid'])[f].min().reset_index().rename(columns={f: f + '_month_min'})
        df4 = tdf.groupby(['userid'])[f].max().reset_index().rename(columns={f: f + '_month_max'})

        action_feature = action_feature.merge(df1, on=['userid'], how='left')
        action_feature = action_feature.merge(df2, on=['userid'], how='left')
        action_feature = action_feature.merge(df3, on=['userid'], how='left')
        action_feature = action_feature.merge(df4, on=['userid'], how='left')

    action_feature = action_feature.fillna(0)

    #每个用户所有记录actionType_coded_1,2,5,6,7,8,9的sum()
    for f in flist:
        tdf = action_fea.groupby(['userid'])[f].sum().reset_index().rename(columns={f: f + '_all_sum'})
        action_feature = action_feature.merge(tdf, on=['userid'], how='left')
    action_feature = action_feature.fillna(0)

    return action_feature

def writeToFile(feature):
    feature.to_csv(feature_url + 'action_tr_feature_v11.csv', sep=',', index=False)

def main():
    action_tr, action_ts=read_data()
    action_feature=build_feature(action_tr)
    writeToFile(action_feature)

if __name__ == '__main__':
    main()