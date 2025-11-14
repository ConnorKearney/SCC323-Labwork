import numpy as np
import activations

def MSE_loss(predicted, actual):
    total = 0
    for i in range(len(predicted)):
        total += (predicted[i] - actual[i])**2
    
    return total/len(predicted)



if __name__ == "__main__":
    U2 = [0,0.1,0.17,0.27]
    A2 = activations.Sigmoid(U2)
    print(A2)
    y_train = [0,1,1,0]

    X_train = np.array([[0,0],[1,0],[0,1],[1,1]])
    print(A2.T)

    print(MSE_loss(U2,y_train))

    print(activations.Sigmoid.d([0.5,0.525,0.542,0.567]))