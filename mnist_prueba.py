# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:29:04 2020

@author: Francis
"""

print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

mnistX, mnistY = fetch_openml("mnist_784",version=1,return_X_y=True)