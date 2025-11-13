import activations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_classes=2,random_state=2005)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

class myPerceptron():
    def __init__(self, n_nodes=2, max_iter=1000, lr=0.1, weights=None, bias=0):
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.lr = lr
        if (not weights or weights.shape != (n_nodes)):
            weights = np.zeros((n_nodes))
        self.weights = weights
        self.bias = bias
        self.activationFunction = activations.Heaviside_Step
        
    
    def fit(self, X, y):
        for i in range(self.max_iter):
            self.runAnotherEpoch(X, y)
            print("Epoch ", i)

    def runAnotherEpoch(self, X, y):
        for i in range(len(X)):
            value = np.sum(np.dot(X[i], self.weights)) + self.bias
            pred_out = self.activationFunction(value)
            
            error = y[i]-pred_out
            self.weights = self.weights + self.lr*error*X[i]
    
    def predict(self, X):
        predicted_output_array = np.zeros((len(X)))
        for i in range(len(X)):
            value = np.sum(np.dot(X[i], self.weights)) + self.bias
            predicted_output_array[i] = self.activationFunction(value)
        return predicted_output_array



P1 = myPerceptron()
P1.fit(X_train, y_train)

def MSE(predicted, actual):
    total = 0
    for i in range(len(predicted)):
        total += (predicted[i] - actual[i])**2
    
    return total/len(predicted)

pred = P1.predict(X_test)
print(MSE(pred,y_test))


