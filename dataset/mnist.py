from tensorflow import keras
from keras.datasets import mnist

def load_mnist(normalize=True, flatten=True, one_hot_label=True):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    if normalize:
        # 数据预处理 归一化到0-1之间
        # 像素值是0-255，我们将其归一化到0-1之间
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    if flatten:
        # 将图像数据展平为一维数组
        x_train = x_train.reshape(-1, 784)  # -1表示自动计算批次大小
        x_test = x_test.reshape(-1, 784)

    if one_hot_label:
        # 将标签数据转换为one-hot编码
        t_train = keras.utils.to_categorical(t_train, 10)
        t_test = keras.utils.to_categorical(t_test, 10)

    return (x_train, t_train), (x_test, t_test)

