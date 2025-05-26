import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from two_layer_net import TwoLayerNet

# 读入mnist数据集
(x_train, t_train), (x_test, t_test) = mnist.load_data()
# 数据预处理 归一化到0-1之间
# 像素值是0-255，我们将其归一化到0-1之间
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 将图像数据展平为一维数组
x_train = x_train.reshape(-1, 784)  # -1表示自动计算批次大小
x_test = x_test.reshape(-1, 784)

# 将标签数据转换为one-hot编码
t_train = keras.utils.to_categorical(t_train, 10)
t_test = keras.utils.to_categorical(t_test, 10)

# 打印数据集形状
print("---------打印数据集形状---------")
print("训练集形状：", x_train.shape)
print("训练集标签形状：", t_train.shape)
print("测试集形状：", x_test.shape)
print("测试集标签形状：", t_test.shape)
print("")

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0 :
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("第{}次 打印train acc : {:.4f}, test acc : {:.4f}".format(i, train_acc, test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()