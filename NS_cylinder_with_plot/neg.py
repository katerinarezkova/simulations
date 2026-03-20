import numpy as np

def neg(y):
    y_neg = (np.absolute(y) - y) / 2
    return y_neg

y = neg(-40)
print(y)