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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print(X_train[:,0])

colours = ['red','blue']
colours_train = [colours[i] for i in y_train]
colours_test = [colours[i] for i in y_test]

#fig, (ax1,ax2) = plt.subplots(2)

#ax1.scatter(X_train[:,0], X_train[:,1],c=colours_train)
#ax1.set_title("Training")

#ax2.scatter(X_test[:,0], X_test[:,1], c=colours_test)
#ax2.set_title("Testing")
#plt.show()


P1 = Perceptron(l1_ratio=0.01, random_state=0)

P1.fit(X_train, y_train)

P1_out = P1.predict(X_test)
print("Perceptron: ", accuracy_score(y_test, P1_out))

###############

MLP1 = MLPClassifier(np.array([5]),max_iter=1000,random_state=0)

MLP1.fit(X_train, y_train)

MLP1_out = MLP1.predict(X_test)
print("MLP: ", accuracy_score(y_test, MLP1_out))

##############


def plot_decision_boundary(clf, X, y, title): 
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1 
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1 
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)) 
 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
 
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm) 
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.coolwarm) 
    plt.title(title) 
    plt.xlabel("Feature 1") 
    plt.ylabel("Feature 2") 
    #plt.show()

    


# perceptron use
plot_decision_boundary(P1, X_train, y_train, "Perceptron")

# mlp use
plot_decision_boundary(MLP1, X_train, y_train, "MLP")
print(MLP1.coefs_)