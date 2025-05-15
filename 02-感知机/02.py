import numpy as np

# 注意，x1, x2是输入信号，只有0/2
# w1， w2是权重
# flag是阈值


def AND(x1, x2):
    w1, w2, flag = 0.5, 0.5, 0.7
    sum = x1 * w1 + x2 * w2
    if sum <= flag:
        return 0
    else:
        return 1
print('----------AND------------')
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
print('----------------------')

# x1 * w1 + x2 * w2, 参考之前numpy对于数组的处理，等价于[x1, x2] * [w1, w2]

def NAND(x1, x2):
    a1 = np.array([x1, x2])
    a2 = np.array([-0.5, -0.5])
    flag = -0.7
    sum = np.sum(a1 * a2)
    if sum <= flag:
        return 0
    else:
        return 1
print('----------NAND------------')
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
print('----------------------')

# 或门
def OR(x1, x2):
    a1 = np.array([x1, x2])
    a2 = np.array([0.5, 0.5])
    flag = 0.2
    sum = np.sum(a1 * a2)
    if sum <= flag:
        return 0
    else:
        return 1

# 前面与门、与非门 如加上一个前置数值，如需要sum + 0.3 <= flag， 则可以实现或门
# 此前置数值称为偏置

# 多层感知机实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
print('----------XOR------------')
print(XOR(0, 0)) # 输出0
print(XOR(1, 0)) # 输出1
print(XOR(0, 1)) # 输出1
print(XOR(1, 1)) # 输出0
print('----------------------')