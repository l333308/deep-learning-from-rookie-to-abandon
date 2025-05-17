import numpy as np
import matplotlib.pylab as plt

#阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int64)

# 效果见阶跃函数.png
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
# plt.show()默认会阻塞，需要手动关闭图像，或传参False，表示不阻塞
plt.show(block=False)
plt.pause(2)  # 暂停，让图形能够显示，然后自动关闭图像

# relu函数
def relu(x):
    return np.maximum(0, x)

# 矩阵乘积
def matrix_dot():
    a1 = np.array([[1, 2, 3], [3, 4, 5]])
    a2 = np.array([[5, 6], [7, 8], [9, 10]])
    return np.dot(a1, a2)

print("矩阵乘法结果：")
print(matrix_dot())

# 恒等函数
def identity_function(x):
    return x

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y