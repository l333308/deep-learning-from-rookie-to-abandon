from cProfile import label
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

# hello
# ! red
# * green
# TODO yellow

def load_mnist_data():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 数据预处理 归一化到0-1之间
    # 像素值是0-255，我们将其归一化到0-1之间
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # 打印数据集形状
    print("---------打印数据集形状---------")
    print("训练集形状：", x_train.shape)
    print("训练集标签形状：", y_train.shape)
    print("测试集形状：", x_test.shape)
    print("测试集标签形状：", y_test.shape)
    print("")

    return (x_train, y_train), (x_test, y_test)

def analyze_single_data(x, y):
    print("---------打印mnist数据 单个---------")
    x = x.reshape(28, 28)
    print("x：", np.shape(x))
    print("x：", x)
    print("y：", y)
    print("")

    return

# 创建一个简单的神经网络模型
def create_model():
    model = keras.Sequential([
        # 将28x28的图像展平为784维向量
        keras.layers.Flatten(input_shape=(28, 28)),
        # 全连接层，128个神经元，使用ReLU激活函数
        keras.layers.Dense(128, activation='relu'),
        # 输出层，10个神经元（对应0-9 10个数字），使用softmax激活函数
        keras.layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_and_evaluate():
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # 创建模型
    model = create_model()

    # 训练模型
    print("\n---------开始训练模型---------")
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # 评估模型
    print("\n---------开始评估模型---------")
    test_loss, tess_acc = model.evaluate(x_test, y_test)
    print(f"测试集准确率: {tess_acc:.4f}")

    # 绘制训练过程中的准确率变化
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label="training accuracy")
    plt.plot(history.history['val_accuracy'], label="validating accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    # 保存模型
    model.save('mnist_model.h5')
    
    return model

def predict_examples(model, x_test, y_test, num_examples=10):
    # 随机选择一些测试样本 进行预测
    indices = np.random.randint(0, x_test.shape[0], num_examples)

    for i, idx in enumerate(indices):
        # 预测结果
        prediction = model.predict(x_test[idx:idx+1])
        predicted_digit = np.argmax(prediction)

        # 使用系统自带的中文字体
        if mpl.get_backend() == 'macosx':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 其他系统使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 显示图像和预测结果
        plt.subplot(1, num_examples, i+1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.title(f'预测：{predicted_digit}\n 实际： {y_test[idx]}')
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    # 训练模型并评估
    model = train_and_evaluate()

    # 加载测试数据
    (_, _), (x_test, y_test) = load_mnist_data()

    # 预测一些示例
    predict_examples(model, x_test, y_test)