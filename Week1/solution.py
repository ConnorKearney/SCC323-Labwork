import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# perceptron work



X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_classes=2,random_state=2005)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2005)

#print(X_train[:,0])

colours = ['red','blue']
colours_train = [colours[i] for i in y_train]
colours_test = [colours[i] for i in y_test]

fig, (ax1,ax2) = plt.subplots(2)

ax1.scatter(X_train[:,0], X_train[:,1],c=colours_train)


ax1.scatter(X_test[:,0], X_test[:,1], c=colours_test)
plt.show
