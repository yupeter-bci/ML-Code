import os

from scipy.ndimage import label

os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP多任务程序, 这里设置为4个线程, 防止出现线程冲突等.
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# 聚类分析用户分
def dm01_聚类分析用户群():
    dataset = pd.read_csv('data/customers.csv')
    dataset.info()
    print('dataset-->\n', dataset)

    X = dataset.iloc[:, [3, 4]]
    print('X-->\n', X)
    mysse = []
    mysscore = []

    # 评估聚类个数
    for i in range(2, 11):
        mykeans = KMeans(n_clusters=i)
        mykeans.fit(X)
        mysse.append(mykeans.inertia_)  # inertia 簇内误差平方和
        ret = mykeans.predict(X)
        mysscore.append(silhouette_score(X, ret))  # sc系数 聚类需要1个以上的类别

    plt.plot(range(2, 11), mysse)
    plt.title('the elbow method')
    plt.xlabel('number of clusters')
    plt.ylabel('mysse')
    plt.grid()
    plt.show()

    plt.title('sh')
    plt.plot(range(2, 11), mysscore)
    plt.grid(True)
    plt.show()

def dm02_聚类分析用户群():
    df = pd.read_csv('data/customers.csv')
    x = df.iloc[:,3:5]
    model = KMeans(n_clusters=5,max_iter=100,random_state=23)
    model.fit(x)
    y_pre = model.predict(x)
    #绘制五个图的散点图
    plt.scatter(x.values[y_pre==0,0],x.values[y_pre==0,1],s=50,c='red',label='0')
    plt.scatter(x.values[y_pre==1,0],x.values[y_pre==1,1],s=50,c='blue',label='1')
    plt.scatter(x.values[y_pre==2,0],x.values[y_pre==2,1],s=50,c='green',label='2')
    plt.scatter(x.values[y_pre==3,0],x.values[y_pre==3,1],s=50,c='yellow',label='3')
    plt.scatter(x.values[y_pre==4,0],x.values[y_pre==4,1],s=50,c='black',label='4')
    plt.legend()
    #显示质心
    plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=200,c='black',marker='*',label='center')
    #设置标题，x轴，y轴
    plt.title('KMeans Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


if __name__ == '__main__':
    # dm01_聚类分析用户群()
    dm02_聚类分析用户群()