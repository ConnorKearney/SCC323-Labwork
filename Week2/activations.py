import numpy as np

def ReLU(x):
    if x > 0: return x
    return 0

def Heaviside_Step(x, activation_value=0):
    if x > activation_value: return 1
    return 0

def Sigmoid(x):
    return (np.e**x)/(np.e**x + 1)

def tanh(x):
    numerator = (np.e**x - np.e**-x)
    denominator = (np.e**x + np.e**-x)
    
    return numerator/denominator

def PReLU(x, alpha=0.1):
    if x > 0: return x
    return (alpha*x)

def LeReLU(x):
    if x > 0: return x
    return (PReLU(x,0.01))

def Para_Sigmoid(x, alpha=1):
    return (1/(1+np.e**(-x/alpha)))

def softsign(x):
    return x/(1+abs(x))


if __name__ == "__main__":
    value = 1
    param = 1
    print(ReLU(value))
    print(Heaviside_Step(value))
    print(Sigmoid(value))
    print(tanh(value))
    print(PReLU(value, param))
    print(LeReLU(value))
    print(Para_Sigmoid(value, param))
    print(softsign(value))
