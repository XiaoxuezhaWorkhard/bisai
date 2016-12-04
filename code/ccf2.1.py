# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ff import feature_fun, other_index
from sklearn.preprocessing import Imputer
gc.enable()
date_min = 0
date_max = 0

def fun(arr):
    ts = pd.DataFrame(np.array(arr[['KWH', 'KWH_READING', 'KWH_READING1']]),index=arr['DATA_DATE'], columns=['KWH', 'KWH_READING', 'KWH_READING1'])
    ts = ts.sort_index()      #按时间排序

    if not ts.index.is_unique:    #处理重复值
        print arr['CONS_NO'].values[0]
        ts = ts.groupby(level=0).sum()      #把每个电表值加起来

    ts = ts.resample('D').mean()   #按天重采样
    #计算功率因素
    PF = (ts['KWH_READING'] - ts['KWH_READING1']) / ts['KWH']
    if sum(PF.notnull())==0:
        PF1 = 1
    else:
        PF1 = np.array(PF[PF.notnull()])[0]

    #处理缺失值这里采用的策略为用缺失天数用电量/缺失天数

    na_days = ts['KWH'].isnull()

    na_days_num = sum(na_days)

    l1 = 0  #第一个非缺失值位置
    l2 = 0  #第二个非缺失值位置
    jiange = 0

    old_i = 0
    mean_values = []
    for n, i in enumerate(np.array(na_days)):

        if i != old_i:
            jiange = jiange + 1
            if jiange == 1:
                l1 = n
            elif jiange == 2:
                jiange == 0
                l2 = n
                mean_na = float(ts['KWH_READING1'].ix[l2] - ts['KWH_READING'].ix[l1-1] )/(l2-l1)
                a = np.full((l2-l1), mean_na)
                mean_values.append(a)
            old_i = i

    mean_values = np.array(mean_values).reshape((1, -1))

    ts['KWH'][na_days] = mean_values

    ts = ts['KWH']
    index = pd.date_range(date_min, date_max)
    ts = ts.reindex(index)   #按天重采样
    ts.fillna(0, inplace= True)

    if sum(ts.notnull())==0:
        print arr['CONS_NO']
        ts = ts.fillna(0.01)
    #特征提取阶段

  #************非差分序列**************
    #按天进行统计
#    mean1,sum1,var1,jizhi1,tra1,trb1,davg_r1,dfou_r1,sd1,bsd1,esd_s1 = feature_fun(ts,120,0)
    #按月每天
#    group_time = ts.groupby(lambda x: x.month)
#    df_month_feature = group_time.apply(feature_fun,12,1)

    #按照月负荷mean_1
    ts_mean = ts.resample('M', how = np.sum)
#    ratio,corr = other_index(ts_mean,median_ser,1)
    mean2,sum2,var2,jizhi2,tra2,trb2,davg_r2,dfou_r2,coef2,sd2,bsd2,esd_s2 = feature_fun(ts_mean,10,2)
    #***********差分序列*****************
 #   ts_diff = ts.diff(1)
 #   ts_diff.dropna(inplace=True)

#    mean1_d, sum1_d, var1_d, jizhi1_d, tra1_d, trb1_d, davg_r1_d, dfou_r1_d, sd1_d, bsd1_d, esd_s1_d = feature_fun(ts_diff,120,0)
    #按月每天
#    group_time = ts_diff.groupby(lambda x: x.month)
#    df_month_feature_diff = group_time.apply(feature_fun,12,1)

    #按照月负荷mean_1
#    ts_mean = ts_diff.resample('M', how = np.sum)
#    ratio_d, corr_d = other_index(ts_mean,median_ser,1)
#    mean2_d, sum2_d, var2_d, jizhi2_d, tra2_d, trb2_d, davg_r2_d, dfou_r2_d, coef2_d, sd2_d, bsd2_d, esd_s2_d = feature_fun(ts_mean,5,2)


    #*******************数据整合************************#
#    daily_feature = np.array([mean1, sum1, var1, jizhi1, tra1, trb1, davg_r1, dfou_r1, sd1, bsd1, esd_s1])

#    month_arr = np.array(df_month_feature.unstack()).ravel()   #月每天

    month_mean = np.array([mean2, sum2, var2, jizhi2, tra2, trb2, davg_r2, dfou_r2, coef2, sd2, bsd2, esd_s2])  #月平均


#    daily_feature_d = np.array([mean1_d, sum1_d, var1_d, jizhi1_d, tra1_d, trb1_d,
#                              davg_r1_d, dfou_r1_d, sd1_d, bsd1_d, esd_s1_d])

 #   month_arr_d = np.array(df_month_feature_diff.unstack()).ravel()   #月每天

#    month_mean_d = np.array([ mean2_d, sum2_d, var2_d, jizhi2_d, tra2_d, trb2_d, davg_r2_d, dfou_r2_d, coef2_d, sd2_d, bsd2_d, esd_s2_d])  #月平均


    if 'LABEL' in arr.columns:
        label = int(pd.unique(arr['LABEL']))
        if label not in [0, 1]:
            print 'fuck'
        data_feature = np.concatenate((month_mean, [PF1], [na_days_num], [label]))
    else:
        data_feature = np.concatenate((month_mean, [PF1], [na_days_num]))
    data_feature = data_feature.reshape((-1, 1))
    del month_mean
    gc.collect()
    return pd.DataFrame(data_feature)

#@profile
def function(data_in, data_out, TRAIN=False):

    data = pd.read_csv(data_in, parse_dates=['DATA_DATE'])
    global date_min, date_max
    date_min = data['DATA_DATE'].min()
    date_max = data['DATA_DATE'].max()
    print('Strat of :' + str(date_min))
    print('End of :' + str(date_max))
    print('Range of :' + str(date_max-date_min))

    if TRAIN:
          #删除全都为0的训练数据
          all_0 = data.groupby('CONS_NO', as_index=False)['KWH_READING1'].sum()
          all_0.columns= 'CONS_NO', 'KWH_READING_SUM'
          data = pd.merge(data, all_0, on='CONS_NO')
          del all_0
          gc.collect()
          data = data[(data['KWH_READING_SUM']!=0)&(data['KWH_READING_SUM'].notnull())]
    data = data.groupby('CONS_NO').apply(fun)
    data = data.unstack()
    data.to_csv(data_out, header=False)
    del data


if __name__ == '__main__':

    train_in = '../data/df_train.csv'
    train_out = '../data/train_fea.csv'
    test_in = '../data/df_test.csv'
    test_out = '../data/test_fea.csv'

#    function(train_in, train_out, True)
    function(test_in, test_out)

