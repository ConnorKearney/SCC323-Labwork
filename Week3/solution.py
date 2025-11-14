from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import activations
import lossFunctions
import numpy as np
import random




class MLP:
    def __init__(self, n_features = 2, n_nodes=3, n_classes=2, lr=0.1, max_iter=1000):
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
        self.weights_1 = np.random.randn(self.n_features, self.n_nodes) * 0.1
        self.weights_2 = W2 = np.random.randn(self.n_nodes, self.n_outputs) * 0.1
        
        self.b1 = np.zeros((1, self.n_nodes))
        self.b2 = np.zeros((1, self.n_outputs))

        print(self.weights_1.shape)
        print(self.weights_2.shape)

        self.internalActivationFunction = activations.ReLU
        self.outputActivationFunction = activations.Sigmoid




    def fit(self, X, y):
        for _ in range(self.max_iter):
            self.nextEpoch(X, y)

    def nextEpoch(self, X, y):
        for i in range(len(X)):
            ## forward pass
            inputs = X[i].reshape(1,2)
            # hidden layer

            hidden_u_values_1 = np.dot(inputs,self.weights_1) + self.b1
            hidden_a_values_1 = self.internalActivationFunction(hidden_u_values_1)

            # output layer

            output_u_values = np.dot(hidden_a_values_1,self.weights_2)  + self.b2
            outputs = self.outputActivationFunction(output_u_values)

            # error
            error = (outputs - y[i])
            
            ## back propogate
            # output layer delta
            d_output = error * self.outputActivationFunction.d(outputs)
            d_weights_2 = np.dot(hidden_a_values_1.T, d_output)
            d_b2 = np.sum(d_output, axis=0,keepdims=True)

            # hidden layer delta
            d_hidden_1 = np.dot(d_output, self.weights_2.T)
            d_u_hidden_1 = d_hidden_1 * self.internalActivationFunction.d(hidden_u_values_1)
            d_weights_1 = np.dot(inputs.T,d_u_hidden_1)
            d_b1 = np.sum(d_u_hidden_1,axis=0, keepdims=True)

            #update weights
            self.weights_2 -= self.lr * d_weights_2
            self.b2 -= self.lr * d_b2
            self.weights_1 -= self.lr * d_weights_1
            self.b1 -= self.lr * d_b1
        
    
    def predict(self, X):
        pass
        


    def predict_single(self, x):
        U1 = np.dot(x, self.weights_1) + self.b1
        A1 = self.internalActivationFunction(U1)
        U2 = np.dot(A1, self.weights_2) + self.b2
        return self.outputActivationFunction(U2)


#X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_classes=2,random_state=2005)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1 Import some data - XOR problem
X_train = np.array([[0,0],[1,0],[0,1],[1,1]])
y_train = np.array([[0],[1],[1],[0]])
X_test = X_train
y_test = y_train

MLP1 = MLP(max_iter=10000)
MLP1.fit(X_train, y_train)

print(MLP1.predict_single(X_train))
print()