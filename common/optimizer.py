import numpy as np

# 随机梯度下降法 Stochastic Gradient Descent
class SGD:
    def __init__(self, lr = 0.01) -> None:
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]
