import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow import keras

def plot_confusion_matrix(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 使用系统自带的中文字体
    if mpl.get_backend() == 'macosx':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 其他系统使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

def analyze_predictions(model, x_test, y_test):
    # 获取预测结果
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, predicted_classes))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, predicted_classes)
    
    # 分析错误预测
    errors = (predicted_classes != y_test)
    error_indices = np.where(errors)[0]
    
    # 显示一些错误预测的例子
    n_errors_to_show = min(5, len(error_indices))
    plt.figure(figsize=(15, 3))
    for i in range(n_errors_to_show):
        idx = error_indices[i]
        plt.subplot(1, n_errors_to_show, i+1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.title(f'预测: {predicted_classes[idx]}\n实际: {y_test[idx]}')
        plt.axis('off')
    plt.suptitle('错误预测示例')
    plt.show()

def visualize_layer_outputs(model, x_test, layer_names=None):
    # 如果没有指定层名称，使用所有层
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers]
    
    # 创建用于获取中间层输出的模型
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # 获取第一个测试样本的激活值
    activations = activation_model.predict(x_test[0:1])
    
    # 可视化每一层的输出
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        if len(layer_activation.shape) == 2:  # 全连接层
            n_features = layer_activation.shape[1]
            n_cols = min(8, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(2*n_cols, 2*n_rows))
            for j in range(n_features):
                plt.subplot(n_rows, n_cols, j+1)
                plt.imshow([[layer_activation[0, j]]], cmap='viridis')
                plt.title(f'特征 {j}')
                plt.axis('off')
            plt.suptitle(f'层 "{layer_name}" 的输出')
            plt.show()

if __name__ == '__main__':
    # 加载之前训练好的模型
    model = keras.models.load_model('mnist_model.h5')
    
    # 加载测试数据
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255
    
    # 分析预测结果
    analyze_predictions(model, x_test, y_test)
    
    # 可视化中间层输出
    visualize_layer_outputs(model)