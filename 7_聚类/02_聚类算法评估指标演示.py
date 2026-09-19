from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score


def dm01_SSE误差平方和求模型参数():
    sse_list = []

    # 产生数据random_state=22固定好
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)

    for clu_num in range(1, 100):
        my_kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=0)
        my_kmeans.fit(x)
        sse_list.append(my_kmeans.inertia_ ) # 获取sse的值

    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sse')
    plt.plot(range(1, 100), sse_list, 'or-')
    plt.show()
    # 通过图像可观察到 n_clusters=4 sse开始下降趋缓, 最佳值4


def dm02_sc误差平方和求模型参数():
    sc_list = []

    # 产生数据random_state=22固定好
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)

    for clu_num in range(2, 100):
        my_kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=0)
        my_kmeans.fit(x)
        y_pre = my_kmeans.predict(x)
        sc_list.append(silhouette_score(x,y_pre)) # 获取sc的值

    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sc')
    plt.plot(range(2, 100), sc_list, 'or-')
    plt.show()

def dm03_ch误差平方和求模型参数():
    ch_list = []

    # 产生数据random_state=22固定好
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)

    for clu_num in range(2, 100):
        my_kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=0)
        my_kmeans.fit(x)
        y_pre = my_kmeans.predict(x)
        ch_list.append(calinski_harabasz_score(x,y_pre)) # 获取ch的值

    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('ch')
    plt.plot(range(2, 100), ch_list, 'or-')
    plt.show()



if __name__ == '__main__':
    # dm01_SSE误差平方和求模型参数()
    # dm02_sc误差平方和求模型参数()
    dm03_ch误差平方和求模型参数()