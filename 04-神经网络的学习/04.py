import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# y = 0.01x^2 + 0.1x
def func_1(x):
    return 0.01*x**2 + 0.1*x

# 以0.1为step 0到20的数组
x = np.arange(0.0, 20.0, 0.1)
y = func_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show(block = False)
plt.pause(2)

print("\n-------------导数-------------------")
print(numerical_diff(func_1, 5))
print(numerical_diff(func_1, 10))

# 求 y = x0^2 + x1^2的偏导数
def func_2(x):
    return x[0]**2 + x[1]**2
    #return np.sum(x**2)

print("\n-------------偏导数-------------------")
def func_tmp1(x0):
    return x0*x0 + 4.0**2
print(numerical_diff(func_tmp1, 3.0))

def func_tmp2(x1):
    return 3.0**2.0 + x1*x1
print(numerical_diff(func_tmp2, 4.0))

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同，值均为zero的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值

    return grad

print("\n-------------梯度-------------------")
print(numerical_gradient(func_2, np.array([3.0, 4.0])))
print(numerical_gradient(func_2, np.array([0.0, 2.0])))
print(numerical_gradient(func_2, np.array([3.0, 0.0])))

print("\n-------------梯度法求func_2最小值-------------------")
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(func_2, init_x = init_x, lr = 0.01, step_num = 100))
print(gradient_descent(func_2, init_x = init_x, lr = 0.1, step_num = 100))
print(gradient_descent(func_2, init_x = init_x, lr = 10.0, step_num = 100))
