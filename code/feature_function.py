# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from datetime import datetime
from sklearn import linear_model

#**********趋势指标*********************
def tendency(ser):
    values = ser.values  #获取序列值
    result =np.sqrt(np.sum(np.square(values)))/len(values)
    return result

def tendency_index(time):
    time_mean = pd.rolling_mean(time,2)   #计算简单窗口平均F序列
    time_cha = time_mean-time     #计算平均与实际值的差
    a_ser = time[time_cha.values>0]   #位于F序列之下的值
    b_ser = time[time_cha.values<0]   #位于F序列之上的值
    tra=tendency(a_ser)
    trb=tendency(b_ser)
    return tra,trb
#**********************************************

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



    #每个用户H个月负荷数据线性拟合斜率
    '''这里需要注意time.index.day'''
  #  if type_n!=0:
   #     reg = linear_model.LinearRegression()    #利用线性回归拟合斜率
   #     reg.fit(np.array(time_axis).reshape((len(time_axis),1)),time.values[::-1])
   #     coef = float(reg.coef_)
   #     return davg_r,dfou_r,coef
  #  else:
    return davg_r,dfou_r

# ******** ****** 波 动性 指标******************
def waviness_index(time,r):
    #每个用户H个月负荷序列的标准差sd
    sd = time.std()
    #前r月负荷序列的标准差bsd_r
    bsd = time[:r].std()
    #后r月负荷序列的标准差esd_r
    esd_s = time[-r:].std()

    return sd,bsd,esd_s

#其他指标

def other_index(time,all_median,r):
    #后r个月平均负荷与所有月平均负荷的比率
    hou_r = time[-r:]
    ratio1 = hou_r.mean()/time.mean()

    #每个用户的负荷序列与所有用户负荷重值序列的相关系数

    corr = time.corr(all_median)

    return ratio1,corr



#time = Series(np.arange(10),
 #             index=pd.date_range('1/1/2000',periods=10))

#tendency_index(time)

#variability_index(time,5)



#将特提取函数放在一起
#参数1：输入时间序列
#参数2：处理类型：0：按天，1：按月每天，2：按月平均，3：按季度每天，4：按季度平均
def feature_fun(time_ser,r,type_n):
    #*******统计指标***************
    if type_n in [0,1,3]:
        time_ser = time_ser.resample('D',fill_method='ffill')
    time_ser = time_ser.sort_index() #时间排序
    mean1 = time_ser.mean()    #计算均值
    sum1 = time_ser.sum()    #总电量
    var1 = time_ser.var()    #方差
    std1 = time_ser.std()    #标准差
    median1 = time_ser.median() #中位数
    max1 = time_ser.max()  #最大值
    min1 = time_ser.min() #最小值
    jizhi = max1-min1 #极值

    #趋势指标
    tra,trb = tendency_index(time_ser)
    #变动性指标
    if type_n==0:
        davg_r,dfou_r = variability_index(time_ser,r,type_n)
    else:
        davg_r,dfou_r = variability_index(time_ser,r,type_n)

    #波动性指标
    sd,bsd,esd_s = waviness_index(time_ser,r)

    #其他指标
#    ratio,corr=other_index(time,r)
    if type_n==0:
        return mean1,sum1,var1,std1,median1,max1,min1,jizhi,tra,trb,davg_r,dfou_r,sd,bsd,esd_s
    elif type_n in [1,3]:
        return pd.DataFrame([[mean1],[sum1],[var1],[std1],[median1],[max1],[min1],[jizhi],[tra],[trb],[davg_r],[dfou_r],[sd],[bsd],[esd_s]])
    else:
        return mean1,sum1,var1,std1,median1,max1,min1,jizhi,tra,trb,davg_r,dfou_r,sd,bsd,esd_s




def mean_f(arr):
    time = np.array(arr['DATE_DATA'])
    kwh = np.array(arr['KWH'])
   # time = [datetime.strptime(x,'%Y/%m/%d') for x in time]
    ts = Series(kwh,index=time)
    ts = ts.fillna(ts.mean())

   # ts = ts.resample('D',fill_method='ffill')

    #按照天进行重采样并补全缺失值 df_daily
  #  df_daily = ts.resample('D',fill_method='ffill')
    mean1,sum1,var1,std1,median1,max1,min1,jizhi1,tra1,trb1,davg_r1,dfou_r1,sd1,bsd1,esd_s1 = feature_fun(ts,100,0)


    #按照12个月每天的数据df_month
    group_time = ts.groupby(lambda x: x.month)
   # df_month = group_time.resample('D',fill_method='ffill')
   # group_time = df_month.groupby(lambda x: x.month)
    df_month_feature = group_time.apply(feature_fun,10,1)

    #按照月平均负荷mean_1
    ts_mean = ts.resample('M',fill_method='ffill')
#    group_time = ts.groupby(lambda x: x.month)
 #   df_month_mean = group_time.mean()
    ratio,corr = other_index(ts_mean,median_ser,1)
    mean2,sum2,var2,std2,median2,max2,min2,jizhi2,tra2,trb2,davg_r2,dfou_r2,sd1,bsd2,esd_s2 = feature_fun(ts_mean,5,2)




    #按照季度采样
    group_time = ts.groupby(lambda x: x.quarter)
   # df_quarter = group_time.resample('D',fill_method='ffill')
    df_quarter_feature = group_time.apply(feature_fun,10,3)


    #每个季度平均
   # group_time = ts.groupby(lambda x: x.quarter)
    df_quarter_mean = ts.resample('Q',fill_method='ffill')
    mean3,sum3,var3,std3,median3,max3,min3,jizhi3,tra3,trb3,davg_r3,dfou_r3,sd3,bsd3,esd_s3 = feature_fun(df_quarter_mean,2,4)


    #*********************************进行数据整合*******************************************

    daily_feature = np.array([mean1,sum1,var1,std1,median1,max1,min1,jizhi1,tra1,trb1,davg_r1,dfou_r1,sd1,bsd1,esd_s1])#每天

    month_arr = np.array(df_month_feature.unstack()).ravel()   #月每天

    month_mean = np.array([ratio,corr,mean2,sum2,var2,std2,median2,max2,min2,jizhi2,tra2,trb2,davg_r2,dfou_r2,sd1,bsd2,esd_s2])  #月平均

    quarter_arr = np.array(df_quarter_feature.unstack()).ravel()  #季度每天

    quarter_mean = np.array([mean3,sum3,var3,std3,median3,max3,min3,jizhi3,tra3,trb3,davg_r3,dfou_r3,sd3,bsd3,esd_s3])   #季度平均

    date_feature = np.concatenate((daily_feature,month_arr,month_mean,quarter_arr,quarter_mean))

    return pd.DataFrame(date_feature)


   # return pd.DataFrame([[mean1],[sum1],[var1],[std1],[median1],[max1],[min1],[jizhi]])

#******feature_select*****
#功能：将原始时间序列数据进行特征提取，并补齐缺失值
#参数：dataIn:输入的数据路径，dataOut:处理完输出的数据路径
dateparse = lambda dates: pd.datetime.strptime(dates,'%Y/%m/%d')

def mean_all(arr):
    time = np.array(arr['DATE_DATA'])
    kwh = np.array(arr['KWH'])
    ts = Series(kwh,index=time)
    group_time = ts.resample('M',fill_method='ffill')
   # df_month_mean = group_time.mean()
    return group_time


def  feature_select(dataIn,dataOut):
    data = pd.read_csv(dataIn,parse_dates=['DATE_DATA'],infer_datetime_format=True,keep_date_col=True,date_parser=dateparse)
    print data.head()

 #   data[data.columns[2]] = data[data.columns[2]].convert_objects(convert_numeric=True)
     #********补全原    有序列的缺失值***********#
    data[data.columns[2]] = data[data.columns[2]].convert_objects(convert_numeric=True)
    grouped1 = data.groupby(data.columns[0])
    def fill_na(x):
        x[data.columns[2]] = x[data.columns[2]].fillna(x[data.columns[2]].mean())
        return x
    transformed1 = grouped1.apply(fill_na)    #补缺失值
    print "补原始序列的缺失值之后："
    print transformed1.head()


    #***********特征提取****************
    grouped2 = transformed1.groupby(data.columns[0])   #以CONS_NO进行分组
    #所有用户负荷中值序列
    grouped_median = data.groupby(data.columns[0])
    cons_month_mean = grouped_median.apply(mean_all)

    cons_month_median = cons_month_mean.unstack()

    global median_ser

    median_ser = Series(cons_month_median.median(),index=cons_month_median.columns)
    #**********************************************************

    transformed2 = grouped2.apply(mean_f)

    result = transformed2.unstack()

   # index = transformed2.index
   # index1 = np.unique([x[0] for x in index])
   # values = np.array(transformed2.ix[:,0]).reshape((len(index1),8))
   # res_arr = np.empty((len(index1),9))
   # res_arr[:,0] = index1
    #res_arr[:,1:] = values

    #*********补全特征提取之后的缺失值************#

   # queshidata = pd.DataFrame(res_arr[:,1:],columns=['mean','sum','var','std','median','max','min','jizhi'],index=res_arr[:,0])
    result = result.fillna(result.mean())
    print result.head()
    np.savetxt(dataOut,result,delimiter=',')
   # result.to_csv(dataOut,header=False)


if __name__ == '__main__':

    traindata = '/usr/local/hadoop/src/data/yichangzhi/traindata.csv'
    trainout = '/usr/local/hadoop/src/data/yichangzhi/trainfeature.csv'
    tstdata = '/usr/local/hadoop/src/data/yichangzhi/testdata.csv'
    tstout = '/usr/local/hadoop/src/data/yichangzhi/testfeature.csv'

    feature_select(traindata,trainout)
    feature_select(tstdata,tstout)







