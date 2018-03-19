#!/usr/bin/python
# -*- coding: utf-8 -*-


import pandas as pd


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score





feature_url='G:/pythonWorkspace/Huangbaoche/feature/'
#merge多个特征表
version='_v11'
user_tr_feature=pd.read_csv(feature_url+'user_tr_feature'+version+'.csv')
action_tr_feature=pd.read_csv(feature_url+'action_tr_feature'+version+'.csv')
orderHistory_tr_feature=pd.read_csv(feature_url+'orderHistory_tr_feature'+version+'.csv')
userComment_tr_feature=pd.read_csv(feature_url+'userComment_tr_feature'+version+'.csv')

train_feature_df=pd.merge(user_tr_feature,action_tr_feature,on='userid',how='left')
train_feature_df=pd.merge(train_feature_df,orderHistory_tr_feature,on='userid',how='left')
train_feature_df=pd.merge(train_feature_df,userComment_tr_feature,on='userid',how='left')
train_feature_df=train_feature_df.fillna(0)

orderFuture_train=pd.read_csv(feature_url+'../data/trainingset/orderFuture_train.csv')
train_totaldf=pd.merge(train_feature_df,orderFuture_train,on='userid',how='left')


#merge多个特征表
user_ts_feature=pd.read_csv(feature_url+'user_ts_feature'+version+'.csv')
action_ts_feature=pd.read_csv(feature_url+'action_ts_feature'+version+'.csv')
orderHistory_ts_feature=pd.read_csv(feature_url+'orderHistory_ts_feature'+version+'.csv')
userComment_ts_feature=pd.read_csv(feature_url+'userComment_ts_feature'+version+'.csv')

test_feature_df=pd.merge(user_ts_feature,action_ts_feature,on='userid',how='left')
test_feature_df=pd.merge(test_feature_df,orderHistory_ts_feature,on='userid',how='left')
test_feature_df=pd.merge(test_feature_df,userComment_ts_feature,on='userid',how='left')
test_feature_df=test_feature_df.fillna(0)

orderFuture_test=pd.read_csv(feature_url+'../data/test/orderFuture_test.csv')
testtotaldf=pd.merge(test_feature_df,orderFuture_test,on='userid',how='left')





train=train_totaldf
train_x=train.drop(['userid','orderType'],axis=1).as_matrix()
train_y=train['orderType'].tolist()


test_x=testtotaldf.drop('userid',axis=1).as_matrix()

dtrain = lgb.Dataset(train_x, train_y)


param={'objective':'binary','metric':'auc'}
gbm = lgb.train(params=param,train_set=dtrain, verbose_eval=100)

pred_lgb_test=gbm.predict(test_x)


datadir='G:/pythonWorkspace/Huangbaoche/data/'
result=pd.DataFrame({'userid':testtotaldf.userid,'orderType':pred_lgb_test},columns=['userid','orderType'])
result.to_csv(datadir+'../result/result_v4.csv',sep=',',header=True,index=False)






