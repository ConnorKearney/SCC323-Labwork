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



plt.plot(X_train[:,0], X_train[:,1],c=y_train[:])
plt.show()
