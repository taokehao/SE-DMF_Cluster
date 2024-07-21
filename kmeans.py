import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    feature.csv聚31类的数据
    '''
    # 设置随机种子
    np.random.seed(42)

    x_data = np.loadtxt('./feature.csv', delimiter=",", dtype="float")
    print("请输入要分的类别数：", end="")
    a = int(input())

    # 创建特征数据进行聚类
    center, _ = kmeans(x_data, a, iter=10, thresh=1e-05, check_finite=True)
    cluster, _ = vq(x_data, center)
    print(cluster)

    print(cluster[103]+1, cluster[117]+1, cluster[131]+1, cluster[145]+1, cluster[159]+1,
          cluster[118]+1, cluster[221]+1, cluster[160]+1, cluster[86]+1, cluster[271]+1)
    print(cluster[133]+1, cluster[232]+1, cluster[188]+1, cluster[290]+1, cluster[71]+1,
    cluster[151]+1, cluster[279]+1, cluster[304]+1, cluster[243]+1, cluster[305]+1, cluster[288]+1, cluster[224]+1)
    # 放308种组合的分布情况
    resultDistri = []
    # 这个循环是将列表扩充到和种类一样大小
    for i in range(a):
        resultDistri.append(0)
    for i in cluster:
        resultDistri[i] += 1
    print("308种组合的分布情况:")
    for i in resultDistri:
        print(i)
