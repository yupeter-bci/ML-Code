import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from pandas.core import col

from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    root_mean_squared_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

class PowerLoadModel:
    def __init__(self):
        logfile_name = 'train' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../',logfile_name).get_logger()
        self.logfile.info("开始创建模型对象")
        self.data_source = data_preprocessing('../data/train.csv')

def ana_data(data):
    """
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    #1.查看数据整体情况
    data = data.copy()
    # data.info()
    #2.负荷整体的分布情况
    fig = plt.figure(figsize=(20,32))
    ax1 = fig.add_subplot(411)
    ax1.hist(data['power_load'],bins=100)
    ax1.set_title('负荷分布直方图')

    #3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    ax2 = fig.add_subplot(412)
    data['hour'] = data['time'].str[11:13]
    data_hour_avg = data.groupby('hour')['power_load'].mean()
    # print(data_hour_avg)
    ax2.plot(data_hour_avg.keys(),data_hour_avg,color='b',linewidth=2)
    ax2.set_title('各个小时的平均负荷趋势图')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('负荷')

    #4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    ax3 = fig.add_subplot(413)
    data['month'] =data['time'].str[5:7]
    data_month_avg = data.groupby('month')['power_load'].mean()
    ax3.plot(data_month_avg.keys(),data_month_avg,color='r',linewidth=2)
    ax3.set_title('各个月份的平均负荷趋势图')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('负荷')

    #5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    data['week_day'] = data['time'].apply(lambda x:pd.to_datetime(x).weekday())
    data['is_workday'] = data['week_day'].apply(lambda x: 1 if x <= 4 else 0)
    power_weekday_mean = data[data['is_workday'] == 1]['power_load'].mean()
    power_weekend_mean = data[data['is_workday'] == 0]['power_load'].mean()
    ax4 = fig.add_subplot(414)
    ax4.bar(['工作日','周末'],[power_weekday_mean,power_weekend_mean],color=['b','b'])
    ax4.set_title('工作日与周末的平均负荷情况')
    ax4.set_xlabel('类型')
    ax4.set_ylabel('负荷')

    plt.savefig('../data/fig/四张图.png')
    plt.show()

def feature_engineering(data, logger):
    """
    对给定的数据源，进行特征工程处理，提取出关键的特征
    1.提取出时间特征：月份、小时
    2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    3.提取昨日同时刻负荷特征
    4.剔除出现空值的样本
    5.整理时间特征，并返回
    :param data: 数据源
    :param logger: 日志
    :return:
    """
    #1.提取出时间特征：月份、小时
    feature_data = data.copy()
    feature_data['hour'] = feature_data['time'].str[11:13]
    feature_data['month'] = feature_data['time'].str[5:7]
    month_hour_data = pd.get_dummies(feature_data[['hour','month']])
    feature_data = pd.concat([feature_data, month_hour_data], axis=1)
    # print(feature_data.head())
    # feature_data.info()
    #2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    load_1h_data = feature_data['power_load'].shift(1)
    load_2h_data = feature_data['power_load'].shift(2)
    load_3h_data = feature_data['power_load'].shift(3)
    load_shift_data = pd.concat([load_1h_data, load_2h_data, load_3h_data], axis=1)
    load_shift_data.columns = ['前1小时', '前2小时', '前3小时']
    feature_data = pd.concat([feature_data, load_shift_data], axis=1)

    #3.提取昨日同时刻负荷特征
    feature_data['yesterday_time'] = feature_data['time'].apply(lambda x:(pd.to_datetime(x)-datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    feature_data['yesterday_load'] = feature_data['yesterday_time'].apply(lambda x: time_load_dict.get(x))
    #4.剔除出现空值的样本
    feature_data.dropna(inplace=True)
    #5.整理时间特征，并返回
    feature_columns = list(month_hour_data.columns) + list(load_shift_data.columns) + ['yesterday_load']
    return feature_columns,feature_data

def model_train(data, features, logger):
    """
    1.数据集切分
    2.网格化搜索与交叉验证
    3.模型实例化
    4.模型训练
    5.模型评价
    6.模型保存
    :param data: 特征工程处理后的输入数据
    :param features: 特征名称
    :param logger: 日志对象
    :return:
    """
    #1.数据集切分
    x = data[features]
    y = data['power_load']
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=23,test_size=0.2)
    #2.网格化搜索与交叉验证
#     logger.info('------ 网格搜索 + 交叉验证 ------')
#     logger.info(f"开始时间：{datetime.datetime.now()}")
#     model1 = XGBRegressor()
#     params = {
#         'n_estimators': [50, 100, 150, 200],
#         'max_depth': [3, 6, 9],
#         'learning_rate': [0.1, 0.01]
# }
#     model = GridSearchCV(model1,params,cv=2)
#     model.fit(x_train,y_train)
#     logger.info(f"结束时间：{datetime.datetime.now()}")
#     logger.info(f"最优参数组合:{model.best_params_}")
    #3.模型实例化
    #4.模型训练
    model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=100)
    model.fit(x_train, y_train)
    #5.模型评价
    y_pre = model.predict(x_test)
    print(f"均方误差:{mean_squared_error(y_test, y_pre)}")
    print(f"平均绝对误差:{mean_absolute_error(y_test, y_pre)}")
    print(f"均方根误差:{root_mean_squared_error(y_test, y_pre)}")
    print(f"平均绝对百分比误差:{mean_absolute_percentage_error(y_test, y_pre)}")

    #6.模型保存
    joblib.dump(model,'../model/xgb_20260813.pkl')
    logger.info("模型保存完成")



if __name__ == '__main__':
    pm = PowerLoadModel()
    # ana_data(pm.data_source)
    feature_columns,feature_data = feature_engineering(pm.data_source,pm.logfile)
    # print(feature_columns)
    # print(feature_data.head(5))
    model_train(feature_data,feature_columns,pm.logfile)
