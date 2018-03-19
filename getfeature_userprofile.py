import pandas as pd
import numpy as np
import datetime

datadir='G:/pythonWorkspace/Huangbaoche/data/'
feature_url='G:/pythonWorkspace/Huangbaoche/feature/'

def read_data():
    userProfile_tr=pd.read_csv(datadir+'trainingset/userProfile_train.csv')
    userProfile_ts=pd.read_csv(datadir+'test/userProfile_test.csv')

    # 对省份编码
    provincedict = {}
    topk = userProfile_tr.province.value_counts()[:10].index.tolist()
    for k in topk:
        provincedict[k] = k
    for k in userProfile_tr.province.value_counts()[10:].index.tolist():
        provincedict[k] = '其他'
    userProfile_tr['province_coded'] = coding(userProfile_tr.province, provincedict)
    userProfile_ts['province_coded'] = coding(userProfile_ts.province, provincedict)
    return userProfile_tr,userProfile_ts

#使用Pandas replace函数定义新函数：
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

def writetoFile(feature):
    feature.to_csv(feature_url + 'user_ts_feature_v11.csv', sep=',', index=False, encoding='utf-8')

def main():
    userProfile_tr, userProfile_ts=read_data()
    user_f = pd.get_dummies(userProfile_ts, columns=['gender', 'province_coded', 'age'],
                            prefix=['gender', 'province_coded', 'age']).drop(['province'], axis=1)
    writetoFile(user_f)

if __name__ == '__main__':
    main()






