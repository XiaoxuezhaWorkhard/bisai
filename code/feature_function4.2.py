# -*- coding: utf-8 -*-

#此版本做了差值计算
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from datetime import datetime
from sklearn import linear_model
import sys


num = 20 #画图的用户数量

#参数：dataIn:输入的数据路径，dataOut:处理完输出的数据路径
dateparse = lambda dates: pd.datetime.strptime(dates,'%Y/%m/%d')

def mean_all(arr):
    time = np.array(arr['DATA_DATE'])
    kwh = np.array(arr['KWH'])
    label = np.unique(np.array(arr['LABEL']))
    ts = Series(kwh,index=time)
    ts = ts.sort_index()
  #  ts = ts.resample('D','mean')
    fft_size = len(ts)
    ts_fft = np.fft.fft(ts) / fft_size
    ts_fft = ts_fft[range(fft_size/2)]
    freq = np.fft.fftfreq(len(ts), d=1/float(fft_size))
    freq = freq[range(fft_size/2)]

#    freq = np.linspace(0, 10, len(ts_fft))
#    ts_fft = np.abs(ts_fft)

#    ts = ts.fillna(ts.mean())
    global num_0,num_1,num
    if label == 0 and num_0<=num:
        num_0 = num_0 + 1
        time_plot1.plot(freq, ts_fft, 'k')
    elif label == 1 and num_1<=num:
        num_1 = num_1 + 1
        time_plot2.plot(freq, ts_fft, 'r')

    if num_0>num and num_1>num:
       # plt.show()
        plt.savefig('../data/fft1.png')
        sys.exit(0)
        print 1
   # df_month_mean = group_time.mean()


plt.clf()

time_plot1 = plt.subplot(211)
time_plot2 = plt.subplot(212)

def  feature_select(dataIn,dataOut):
    reader = pd.read_csv(dataIn, parse_dates=['DATA_DATE'], infer_datetime_format=True, keep_date_col=True, date_parser=dateparse, iterator = True)
    loop = 1
    chunkSize = 8000000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
            loop = loop+1
        except StopIteration:
            loop = False
            print 'Iteration is stopped.'

    data = pd.concat(chunks, ignore_index = True)
    del chunks

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







