# -*- coding: utf-8 -*-

#此版本做了差值计算
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from datetime import datetime
from sklearn import linear_model
import sys
#*****************变动性指标*******************
#参数time:输入时间序列
#参数r:前r个月与后 r个月
def variability_index(time,r,type_n):
    qian_r = time[:r]
    hou_r = time[-r:]

    davg_r=np.sum(qian_r.values)/r - np.sum(hou_r.values)/r  #前r个月与后r个月平均负荷的差值

    #前r个月与后r个月离散傅里叶变换的系数序列的差值序列的模

    qian_fft = np.fft.fft(qian_r)   #前r个月的数据进行FFT变换
    hou_fft = np.fft.fft(hou_r)    #后r个月的

  #  if type_n!=4:
  #      num = int(r/2-1)
  #  else:
    num = r

    Yn1 = qian_fft[:num]
    Yn2 = hou_fft[-num:]
    Yn2 = Yn2[::-1]    #倒序
    a = Yn1-Yn2
    dfou_r = np.sqrt(sum((a.conjugate()*a).real))
    if type_n in [1,3]:
        time_axis = time.index.day
    elif type_n == 2:
        time_axis = time.index.month
    elif type_n == 4:
        time_axis = time.index.quarter
    elif type_n == 5:
        time_axis = time.index



    #每个用户H个月负荷数据线性拟合斜率
    '''这里需要注意time.index.day'''
    if type_n!=0:
        reg = linear_model.LinearRegression()    #利用线性回归拟合斜率
        reg.fit(np.array(time_axis).reshape((len(time_axis),1)),time.values[::-1])
        coef = float(reg.coef_)
        return davg_r,dfou_r,coef
    else:
        return davg_r,dfou_r


#参数：dataIn:输入的数据路径，dataOut:处理完输出的数据路径
dateparse = lambda dates: pd.datetime.strptime(dates,'%Y/%m/%d')

def mean_all(arr):
    time = np.array(arr['DATA_DATE'])
    kwh = np.array(arr['KWH'])
    label = np.unique(np.array(arr['LABEL']))
    ts = Series(kwh,index=time)
    ts = ts.resample('D')
#    ts = ts.fillna(ts.mean())
    global num_0,num_1
    if label == 0 and num_0<=10:
        num_0 = num_0 + 1
        time_plot.plot(ts.index,ts.values,'k--')
    if label == 1 and num_1<=10:
        num_1 = num_1 + 1
        time_plot.plot(ts.index,ts.values,color='r')

    if num_0>10 and num_1>10:
       # plt.show()
        plt.savefig('../data/time.png')
        sys.exit(0)
        print 1
   # df_month_mean = group_time.mean()


time_plot = plt.subplot(211)

def  feature_select(dataIn,dataOut):
    data = pd.read_csv(dataIn,parse_dates=['DATA_DATE'],infer_datetime_format=True,keep_date_col=True,date_parser=dateparse)
    print data.head()

 #   data[data.columns[2]] = data[data.columns[2]].convert_objects(convert_numeric=True)
     #********补全原    有序列的缺失值***********#
    data[data.columns[2]] = data[data.columns[2]].convert_objects(convert_numeric=True)
    #**********************************************************
#   time_plot = plt.subplot(211)
    grouped2 = data.groupby(data.columns[0])
    transformed2 = grouped2.apply(mean_all)

if __name__ == '__main__':

    traindata = '../data/df_train.csv'
    trainout = '../data/df_test.csv'
    num_0 = 0
    num_1 = 0
    feature_select(traindata,trainout)







