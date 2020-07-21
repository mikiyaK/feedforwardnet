#util.py
import numpy as np

def cross_entropy(prob, t):
    return -np.mean(np.log(prob[np.arange(prob.shape[0]), t] + 1e-7))

def softmax(y):
    y = y - np.max(y, axis=1, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

def softmax_cross_entropy(y, t):
    return cross_entropy(softmax(y), t)
