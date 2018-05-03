import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost as cb
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 600)

import os
print(os.listdir("orig_data"))

testing = pd.read_csv('orig_data/poker-hand-testing.data',names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','hand'])
training = pd.read_csv('orig_data/poker-hand-training-true.data',names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','hand'])

print("testing header\n",testing.head())
print("training header\n",training.head())

print("training shape",training.shape)
print("testing shape",testing.shape)

X = training.drop(['hand'],axis=1)
y = training.hand
Xte = testing.drop(['hand'],axis=1)
yte = testing.hand
