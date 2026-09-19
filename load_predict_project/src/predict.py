import os
import pandas as pd
import numpy as np
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


class PowerLoadPredict(object):
    def __init__(self,file_path):
        logfile_name = 'train' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        self.data_source = data_preprocessing(file_path)
        self.time_load_dict = self.data_source.set_index('time')['power_load'].to_dict()

def pred_feature_extract(data_dict, time, logger):
    """
    预测数据解析特征，保持与模型训练时的特征列名一致
    1.解析时间特征
    2.解析时间窗口特征
    3.解析昨日同时刻特征
    :param data_dict:历史数据，字典格式，key：时间，value:负荷
    :param time:预测时间，字符串类型，格式为2024-12-20 09:00:00
    :param logger:日志对象
    :return:
    """
    logger.info(f'=========解析预测时间为：{time}所对应的特征==============')
    # 特征列清单
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     '前1小时', '前2小时', '前3小时', 'yesterday_load']
    # 小时特征数据，使用列表保存起来
    pre_time = time[11:13]
    hour_list = []
    for i in range(24):
        if pre_time==feature_names[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)
    pre_month = time[5:7]
    month_list = []
    for i in range(24,36):
        if pre_month==feature_names[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)
    last_1h_time = (pd.to_datetime(time)-pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time, 500)
    last_2h_time = (pd.to_datetime(time)-pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 500)
    last_3h_time = (pd.to_datetime(time)-pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 500)

    yesterday_time = (pd.to_datetime(time)-pd.to_timedelta('1D')).strftime('%Y-%m-%d %H:%M:%S')
    yesterday_load = data_dict.get(yesterday_time, 500)
    feature_data = hour_list + month_list + [last_1h_load, last_2h_load, last_3h_load, yesterday_load]
    feature_df = pd.DataFrame([feature_data], columns=feature_names)
    return feature_df

def prediction_plot(data):
    """
    绘制时间与预测负荷折线图，时间与真实负荷折线图，展示预测效果
    :param data: 数据一共有三列：时间、真实值、预测值
    :return:
    """
    # 绘制在新数据
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot()
    # 绘制时间与真实负荷的折线图
    ax.plot(data['预测时间'], data['真实负荷'], label='真实负荷')
    # 绘制时间与预测负荷的折线图
    ax.plot(data['预测时间'], data['预测负荷'], label='预测负荷')
    ax.set_ylabel('负荷')
    ax.set_xlabel('时间')
    ax.set_title('预测负荷以及真实负荷的折线图')
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('../data/fig/预测结果.png')
    plt.show()

if __name__ == '__main__':
    plp = PowerLoadPredict('../data/test.csv')
    # print(plp.data_source)
    model = joblib.load('../model/xgb_20260813.pkl')
    pre_times = plp.data_source['time'][plp.data_source['time']>='2015-08-01 00:00:00']

    evaluate_list = []
    for pre_time in pre_times:
        # print(f"正在预测{pre_time}时间的负荷--")
        time_load_dict_masked = {k:v for k,v in plp.time_load_dict.items() if k < pre_time}
        feature_df = pred_feature_extract(plp.time_load_dict,pre_time,plp.logfile)
        y_pre = model.predict(feature_df)
        true_load = plp.time_load_dict[pre_time]
        evaluate_list.append([pre_time,true_load,y_pre[0]])

    evaluate_df = pd.DataFrame(evaluate_list,columns=['预测时间','真实负荷','预测负荷'])
    # print(evaluate_df)
    prediction_plot(evaluate_df)