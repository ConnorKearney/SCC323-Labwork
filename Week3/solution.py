from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import activations
import lossFunctions
import numpy as np
import random




class MLP:
    def __init__(self, n_features = 2, n_nodes=3, n_classes=2, lr=0.1, max_iter=1000):
        random.seed(2005)
        self.n_features = n_features
        self.n_nodes = n_nodes
        if n_classes > 2:
            self.n_outputs = n_classes
        else:
            self.n_outputs = 1

        self.n_classes = n_classes
        self.lr = lr
        self.max_iter = max_iter

        random_values = [random.random() for _ in range(1000)]
        self.weights_1 = np.array(random_values[:n_features*n_nodes]).reshape((n_features, n_nodes))
        self.weights_2 = np.array(random_values[n_features*n_nodes:n_features*n_nodes + n_nodes*self.n_outputs]).reshape((n_nodes, self.n_outputs))
        

        print(self.weights_1.shape)
        print(self.weights_2.shape)

        self.internalActivationFunction = activations.ReLU
        self.outputActivationFunction = activations.Sigmoid




    def fit(self, X, y):
        for _ in range(self.max_iter):
            self.nextEpoch(X, y)

    def nextEpoch(self, X, y):
        for i in range(1):
            # forward pass
            inputs = X[i]
            internal_values = np.dot(inputs,self.weights_1)
            #print(internal_values)
            internal_values_activated = np.array(list(map(self.internalActivationFunction,internal_values)))
            ##print(internal_values_activated)
            outputs = np.array(list(map(self.outputActivationFunction,np.dot(internal_values_activated,self.weights_2))))
            #print(outputs)

            error = y[i] - outputs
            #print(error)

            # back propogate
            delta_1 = error * self.outputActivationFunction.d(np.sum(np.dot(internal_values_activated,self.weights_2)))
            #print(delta_1)
            self.weights_2 = self.weights_2 + self.lr * delta_1 * internal_values_activated
            #print(self.weights_2)
            delta_2 = delta_1 * self.weights_2 * np.array(list(map(self.internalActivationFunction.d, np.cross(inputs, self.weights_1))))
            print(delta_2.shape)
            self.weights_1 = self.weights_1 + self.lr * delta_2 * inputs

        
    
    def predict(self):
        pass


X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_classes=2,random_state=2005)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

MLP1 = MLP(max_iter=1)
MLP1.fit(X_train, y_train)
#print(MLP1.weights_1)
#print(MLP1.weights_2)