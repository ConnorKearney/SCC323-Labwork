import numpy as np

class ReLU_func:
    def __call__(self, x):
        if x > 0: return x
        return 0
    
    def d(self, x):
        if x > 0: return 1
        return 0

class Sigmoid_func:
    def __call__(self, x):
        return (np.e**x)/(np.e**x + 1)
    
    def d(self, x):
        value = self(x)
        return value * (1-value)
    
class tanh_func:
    def __call__(self, x):
        numerator = (np.e**x - np.e**-x)
        denominator = (np.e**x + np.e**-x)
    
        return numerator/denominator
    
    def d(self, x):
        return 1-self(x)**2
    



def Heaviside_Step(x, activation_value=0):
    if x > activation_value: return 1
    return 0

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


# Derivatives

def d_SiLU(x):
    return Sigmoid(x) + x*d_Sigmoid(x)


ReLU = ReLU_func()
Sigmoid = Sigmoid_func()
tanh = tanh_func()


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
