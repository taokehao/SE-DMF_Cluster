# -*- encoding:utf-8 -*-
from mendeleev import element
import numpy as np
import math
import csv
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


def log10(x):
    '''用来计算相应数值的log(x)
    :param x: 输入值
    :return: 输出log(x)
    '''
    return math.log10(x)


def sigmoid(x):
    '''用来计算函数的sigmoid值
    :param x: 输入值
    :return: 输出sigmoid值
    '''
    return 1 / (1 + math.exp(-x))


def featue(newRow, i, j):
    '''用来记录第一聚类使用的特征
    :param newRow: 新的一行列表
           i: 元素名
           j: 元素名
    :return: 返回填充满的列表
    '''
    newRow.append(log10(abs(element(i).atomic_number + element(j).atomic_number)))
    newRow.append(log10(abs(element(i).dipole_polarizability + element(j).dipole_polarizability)))
    newRow.append(log10(abs(element(i).covalent_radius + element(j).covalent_radius)))
    newRow.append(log10(abs(element(i).en_pauling + element(j).en_pauling)))
    newRow.append(log10(abs(element(i).group_id + element(j).group_id)))
    newRow.append(log10(abs(element(i).ionenergies[1] + element(j).ionenergies[1])))
    # newRow.append(math.log10(abs(element(i).atomic_weight + element(j).atomic_weight)))
    # newRow.append(math.log10(abs(element(i).atomic_volume + element(j).atomic_volume)))
    # newRow.append(math.log10(abs(element(i).abundance_crust + element(j).abundance_crust)))
    # newRow.append(math.log10(abs(element(i).boiling_point + element(j).boiling_point)))

    # -
    newRow.append(abs(element(i).atomic_number - element(j).atomic_number))
    newRow.append(abs(element(i).dipole_polarizability - element(j).dipole_polarizability))
    newRow.append(abs(element(i).covalent_radius - element(j).covalent_radius))
    newRow.append(abs(element(i).en_pauling - element(j).en_pauling))
    newRow.append(abs(element(i).group_id - element(j).group_id))
    newRow.append(abs(element(i).ionenergies[1] - element(j).ionenergies[1]))
    # newRow.append(abs(element(i).atomic_weight - element(j).atomic_weight))
    # newRow.append(abs(element(i).atomic_volume - element(j).atomic_volume))
    # newRow.append(abs(element(i).abundance_crust - element(j).abundance_crust))
    # newRow.append(abs(element(i).boiling_point - element(j).boiling_point))

    # *
    newRow.append(log10(abs(element(i).atomic_number * element(j).atomic_number)))
    newRow.append(log10(abs(element(i).dipole_polarizability * element(j).dipole_polarizability)))
    newRow.append(log10(abs(element(i).covalent_radius * element(j).covalent_radius)))
    newRow.append(log10(abs(element(i).en_pauling * element(j).en_pauling)))
    newRow.append(log10(abs(element(i).group_id * element(j).group_id)))
    newRow.append(log10(abs(element(i).ionenergies[1] * element(j).ionenergies[1])))
    # newRow.append(math.log10(abs(element(i).atomic_weight * element(j).atomic_weight)))
    # newRow.append(math.log10(abs(element(i).atomic_volume * element(j).atomic_volume)))
    # newRow.append(math.log10(abs(element(i).abundance_crust * element(j).abundance_crust)))
    # newRow.append(math.log10(abs(element(i).boiling_point * element(j).boiling_point)))

    # /
    newRow.append(log10(abs(element(i).atomic_number / element(j).atomic_number)))
    newRow.append(log10(abs(element(i).dipole_polarizability / element(j).dipole_polarizability)))
    newRow.append(log10(abs(element(i).covalent_radius / element(j).covalent_radius)))
    newRow.append(log10(abs(element(i).en_pauling / element(j).en_pauling)))
    newRow.append(log10(abs(element(i).group_id / element(j).group_id)))
    newRow.append(log10(abs(element(i).ionenergies[1] / element(j).ionenergies[1])))
    # newRow.append(math.log10(abs(element(i).atomic_weight / element(j).atomic_weight)))
    # newRow.append(math.log10(abs(element(i).atomic_volume / element(j).atomic_volume)))
    # newRow.append(math.log10(abs(element(i).abundance_crust / element(j).abundance_crust)))
    # newRow.append(math.log10(abs(element(i).boiling_point / element(j).boiling_point)))
    return newRow

def featue3(newRow, i, j):
    '''开始记录特征的生成方式
    :param newRow: 新的一行列表
           i: 元素名
           j: 元素名
    :return: 返回填充满的列表
    '''
    # +
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability + element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius + element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling + element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id + element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] + element(j).ionenergies[1]))
    # newRow.append(math.log10(abs(element(i).atomic_weight + element(j).atomic_weight)))
    # newRow.append(math.log10(abs(element(i).atomic_volume + element(j).atomic_volume)))
    # newRow.append(math.log10(abs(element(i).abundance_crust + element(j).abundance_crust)))
    # newRow.append(math.log10(abs(element(i).boiling_point + element(j).boiling_point)))

    # -
    newRow.append(sigmoid(element(i).atomic_number - element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability - element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius - element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling - element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id - element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] - element(j).ionenergies[1]))
    # newRow.append(abs(element(i).atomic_weight - element(j).atomic_weight))
    # newRow.append(abs(element(i).atomic_volume - element(j).atomic_volume))
    # newRow.append(abs(element(i).abundance_crust - element(j).abundance_crust))
    # newRow.append(abs(element(i).boiling_point - element(j).boiling_point))

    # *
    newRow.append(sigmoid(element(i).atomic_number * element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability * element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius * element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling * element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id * element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] * element(j).ionenergies[1]))
    # newRow.append(math.log10(abs(element(i).atomic_weight * element(j).atomic_weight)))
    # newRow.append(math.log10(abs(element(i).atomic_volume * element(j).atomic_volume)))
    # newRow.append(math.log10(abs(element(i).abundance_crust * element(j).abundance_crust)))
    # newRow.append(math.log10(abs(element(i).boiling_point * element(j).boiling_point)))

    # /
    newRow.append(sigmoid(element(i).atomic_number / element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability / element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius / element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling / element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id / element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] / element(j).ionenergies[1]))
    # newRow.append(math.log10(abs(element(i).atomic_weight / element(j).atomic_weight)))
    # newRow.append(math.log10(abs(element(i).atomic_volume / element(j).atomic_volume)))
    # newRow.append(math.log10(abs(element(i).abundance_crust / element(j).abundance_crust)))
    # newRow.append(math.log10(abs(element(i).boiling_point / element(j).boiling_point)))
    return newRow

def featue4(newRow, i, j):
    '''开始记录特征的生成方式
    :param newRow: 新的一行列表
           i: 元素名
           j: 元素名
    :return: 返回填充满的列表
    '''
    # +
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability + element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius + element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling + element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id + element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] + element(j).ionenergies[1]))
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number + element(i).dipole_polarizability + element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number + element(i).covalent_radius + element(j).covalent_radius))
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number + element(i).en_pauling + element(j).en_pauling))
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number + element(i).group_id + element(j).group_id))
    newRow.append(sigmoid(element(i).atomic_number + element(j).atomic_number + element(i).ionenergies[1] + element(j).ionenergies[1]))
    newRow.append(sigmoid(element(i).dipole_polarizability + element(j).dipole_polarizability + element(i).covalent_radius + element(j).covalent_radius))


    # -
    newRow.append(sigmoid(element(i).atomic_number - element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability - element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius - element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling - element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id - element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] - element(j).ionenergies[1]))

    # *
    newRow.append(sigmoid(element(i).atomic_number * element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability * element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius * element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling * element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id * element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] * element(j).ionenergies[1]))


    # /
    newRow.append(sigmoid(element(i).atomic_number / element(j).atomic_number))
    newRow.append(sigmoid(element(i).dipole_polarizability / element(j).dipole_polarizability))
    newRow.append(sigmoid(element(i).covalent_radius / element(j).covalent_radius))
    newRow.append(sigmoid(element(i).en_pauling / element(j).en_pauling))
    newRow.append(sigmoid(element(i).group_id / element(j).group_id))
    newRow.append(sigmoid(element(i).ionenergies[1] / element(j).ionenergies[1]))

    return newRow

def data_export():
    '''
    生成聚类特征的函数
    结果输出在featureX.csv中
    '''
    element_M = ['Be', 'Mg', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Sr', 'Pd', 'Ag',
                 'Cd', 'Ba', 'Pt', 'Hg', 'Pb']
    element_Ni = ['Be', 'Mg', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Cd', 'Hg', 'Pt', 'Pd', 'Ag', 'Pb']
    allRow = []
    for i in element_M:
        for j in element_Ni:
            newRow = []
            print(i, j)
            # +
            newRow = featue(newRow, i, j)
            allRow.append(newRow)
    for i in allRow:
        csvFile = open('./feature.csv', 'a', newline='', encoding='utf-8')
        writer = csv.writer(csvFile)
        writer.writerow(i)  # 数据写入文件中zz
        csvFile.close()


if __name__ == '__main__':
    # data_export()
    '''
    feature.csv聚31类的数据
    '''
    x_data = np.loadtxt('./feature.csv', delimiter=",", dtype="float")
    print("请输入要分的类别数：", end="")
    a = input()
    # 创建特征数据进行聚类

    # 下面是画图
    plt.figure(figsize=(20, 6), linewidth=5.0)
    Z = linkage(x_data, method='average', metric='euclidean')
    f = fcluster(Z, t=int(a), criterion='maxclust')  # 聚类，这里t阈值的选择很重要
    print(f)  # 打印类标签
    p = dendrogram(Z, color_threshold=20.3125)
    #plt.show()
    # 放308种组合的分布情况
    resultDistri = []
    # 这个循环是将列表扩充到和种类一样大小
    for i in range(int(a)):
        resultDistri.append(0)
    # 这个循环是将统计308种组合的分布情况
    print(f[131], f[103], f[117], f[145], f[159])
    for i in f:
        resultDistri[i - 1] += 1
    print("308种组合的分布情况:")
    print(resultDistri)
    # 记录那49个都是什么结构
    # index = 1
    # for i in f:
    #     if i == 3:
    #         readlines = open("./material-name.txt", "r").readlines()
    #         print(readlines[index - 1])
    #     index += 1
    #
    # index = 1
    # count_49 = 0
    # count_46 = 5
    # count_36 = 10
    # for i in f:
    #     newRow = []
    #     if i == 3:
    #         csvFile = open("./data/49-50.csv", 'a', newline='', encoding='utf-8')
    #         writer = csv.writer(csvFile)
    #         for item in x_data[index-1]:
    #             newRow.append(float(item) + count_49 * 50)
    #         writer.writerow(newRow)  # 数据写入文件中zz
    #         count_49 += 1
    #     if i == 6:
    #         csvFile = open("./data/46-50.csv", 'a', newline='', encoding='utf-8')
    #         writer = csv.writer(csvFile)
    #         for item in x_data[index - 1]:
    #             newRow.append(float(item) + count_46 * 50)
    #         writer.writerow(newRow)  # 数据写入文件中zz
    #         count_46 += 1
    #     if i == 16:
    #         csvFile = open("./data/36-50.csv", 'a', newline='', encoding='utf-8')
    #         writer = csv.writer(csvFile)
    #         for item in x_data[index - 1]:
    #             newRow.append(float(item) + count_36 * 50)
    #         writer.writerow(newRow)  # 数据写入文件中zz
    #         count_36 += 1
    #     index += 1
    for i in resultDistri:
        print(i)