# -*- encoding:utf-8 -*-
from mendeleev import element
import numpy as np
import math
import csv
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


if __name__ == '__main__':
    element_M = ['Be', 'Mg', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Sr', 'Pd', 'Ag',
                 'Cd', 'Ba', 'Pt', 'Hg', 'Pb']
    element_Ni = ['Be', 'Mg', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Cd', 'Hg', 'Pt', 'Pd', 'Ag', 'Pb']
    for i in element_Ni:
        if i in element_M:
            print("'" + str(i) + "', ", end="")
