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


#merge多个特征表  -----------test
# user_ts_feature=pd.read_csv(feature_url+'user_ts_feature.csv')
# action_ts_feature=pd.read_csv(feature_url+'action_ts_feature.csv')
# orderHistory_ts_feature=pd.read_csv(feature_url+'orderHistory_ts_feature.csv')
# userComment_ts_feature=pd.read_csv(feature_url+'userComment_ts_feature.csv')
#
# test_feature_df=pd.merge(user_ts_feature,action_ts_feature,on='userid',how='left')
# test_feature_df=pd.merge(test_feature_df,orderHistory_ts_feature,on='userid',how='left')
# test_feature_df=pd.merge(test_feature_df,userComment_ts_feature,on='userid',how='left')
# test_feature_df=test_feature_df.fillna(0)
#
# orderFuture_test=pd.read_csv(feature_url+'../data/test/orderFuture_test.csv')
# testtotaldf=pd.merge(test_feature_df,orderFuture_test,on='userid',how='left')


#划分线下训练集和测试集

data=train_totaldf.drop('userid',axis=1)
X=data.drop('orderType',axis=1).as_matrix()
y=data['orderType']


X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(X, y, test_size=0.3, random_state=42)


dtrain = lgb.Dataset(data=X_train_local, label=y_train_local)
dtest_local = lgb.Dataset(data=X_test_local,label=y_test_local)


#param = {'num_leaves':7,'num_boost_round':1000, 'objective':'binary','metric':'auc',"learning_rate" : 0.05, "boosting":"gbdt"}
param={'objective':'binary','metric':'auc','learning_rate' : 0.05}
gbm = lgb.train(params=param,train_set=dtrain, verbose_eval=100)
#gbm.save_model('model/lgb.txt')
pred_lgb_test_local = gbm.predict(X_test_local)
pred_lgb_train_local = gbm.predict(X_train_local)
print('train_local auc: %g' % roc_auc_score(y_true=y_train_local,y_score=pred_lgb_train_local))
print('test_local auc: %g' % roc_auc_score(y_true=y_test_local,y_score=pred_lgb_test_local))


imp = gbm.feature_importance()
f_imp=pd.DataFrame({'imp':imp,'f_name':data.drop('orderType',axis=1).columns}).sort_values(by=['imp'],ascending=False)
#f_imp.to_csv('feature/f_imp_v1.txt',index=False,encoding='utf-8')
print(f_imp.head(10))









