import pandas as pd
import numpy as np

def data_preprocessing(path):
    """
    1.获取数据源
    2.时间格式化，转为2024-12-20 09:00:00这种格式
    3.按时间升序排序
    4.去重
    :param path:
    :return:
    """
    # 1.获取数据源
    data = pd.read_csv(path)
    # 2.时间格式化
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # 3.按时间升序排序
    data.sort_values(by='time', inplace=True)
    # 4.去重
    data.drop_duplicates(inplace=True)
    # print(data.head())
    return data

if __name__ == '__main__':
    data_preprocessing()