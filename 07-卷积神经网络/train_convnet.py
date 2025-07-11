import sys, os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 添加通道维度
x_train = x_train.reshape(-1, 1, 28, 28)  # 将形状从(批量大小, 高度, 宽度)改为(批量大小, 通道数, 高度, 宽度)
x_test = x_test.reshape(-1, 1, 28, 28)

# 处理花费时间较长的情况下减少数据 
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim = (1, 28, 28),
conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
hidden_size = 100, output_size = 10, weight_init_std = 0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
epochs = max_epochs, mini_batch_size = 100,
optimizer = 'Adam', optimizer_param = {'lr': 0.001},
evaluate_sample_num_per_epoch = 1000)
trainer.train()

# 保存参数
network.save_params("params.pk1")
print("Saved Network Parameters!")

# 绘制图像
makers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


