import pandas as pd
import numpy as np
import datetime

datadir='G:/pythonWorkspace/Huangbaoche/data/'
feature_url='G:/pythonWorkspace/Huangbaoche/feature/'

def read_data():

    userComment_tr = pd.read_csv(datadir + 'trainingset/userComment_train.csv')
    userComment_ts = pd.read_csv(datadir + 'test/userComment_test.csv')

    return userComment_tr,userComment_ts


def build_feature(userComment):
    userComment = userComment.fillna(-999)
    userComment_feature = userComment[['userid', 'rating']]
    #用户tag个数
    userComment_feature['tag_num'] = userComment.tags.apply(lambda x: 0 if x == -999 else len(x.split('|')))
    #用户comment关键字个数
    userComment_feature['comment_key_num'] = userComment.commentsKeyWords.apply(
        lambda x: 0 if x == -999 else len(x.split(',')))
    return userComment_feature

def writeToFile(feature):
    feature.to_csv(feature_url + 'userComment_tr_feature_v11.csv', sep=',', index=False, encoding='utf-8')

def main():
    userComment_tr, userComment_ts=read_data()
    userComment_feature=build_feature(userComment_tr)
    writeToFile(userComment_feature)

if __name__ == '__main__':
    main()