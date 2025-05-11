py基础，重点在numpy、Matplotlib的使用。
深度学习，经常用到数组、矩阵运算，用numpy包实现。
图形绘制 / 数据可视化，用Matplotlib包实现。

# numpy

## 数组运算
两个长度相同的数组可以进行运算。
```python
➜ python3
Python 3.11.12
>>> import numpy as np
>>> x = np.array([1, 2, 3])
>>> print(x)
[1 2 3]
>>> type(x)
<class 'numpy.ndarray'>
>>> y = np.array([2,4,6])
>>> x + y
array([3, 6, 9])
>>> x - y
array([-1, -2, -3])
>>> x * y
array([ 2,  8, 18])
>>> x / y
array([0.5, 0.5, 0.5])

```

接上面，把x的长度改为4，y的长度还是3。可见长度不同时，无法进行运算。
```python
>>> x = np.array([1,2,3,4])
>>> x + y
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (4,) (3,)
>>> x - y
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (4,) (3,)
>>> x * y
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (4,) (3,)
>>> x / 2
array([0.5, 1. , 1.5, 2. ])
>>> x = np.array([1,2], [3, 4])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Field elements must be 2- or 3-tuples, got '3'
```

## 广播
上面可见，当数组长度不同(3 != 4)时，无法进行运算。
但是有个特殊情况，长分别为n、1时，可以进行运算。会把长度为1，元素值为v的数组复制n份，然后再进行运算。
如[1, 2, 3] + [10] = [11, 12, 13], 此时[10]会拓宽为[10, 10, 10]。
```python
>>> x = np.array([1,2,3])
>>> y = np.array([10])
>>> x + y
array([11, 12, 13])
>>> x - y
array([-9, -8, -7])
>>> x * y
array([10, 20, 30])
```
如y不是一维数组，而是标量，效果和一维数组一样。

## 访问数组元素
用索引（下标）访问。
```python
>>> x = np.array([1,2,3])
>>> x[0]
1
```
或遍历获取
```python
>>> for i in x:
...     print(i)
...
1
2
3
```

# Matplotlib
## sin曲线
代码运行效果看plt-sin.png
```python
✗ python3
Python 3.11.12 (main, Apr  8 2025, 14:15:29) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> x = np.arange(0,6,0.1)
>>> y = np.sin(x)
>>> plt.plot(x,y)
[<matplotlib.lines.Line2D object at 0x106202350>]
>>> plt.show()
2025-05-11 12:18:21.473 Python[61035:5247139] +[IMKClient subclass]: chose IMKClient_Legacy
2025-05-11 12:18:21.473 Python[61035:5247139] +[IMKInputSession subclass]: chose IMKInputSession_Legacy
```

## sin + cos曲线
接下来把cos也加到图里，再完善一下。plt-sin+cos.png
```python
>>> x = np.arange(0, 6, 0.1)
>>> y1 = np.sin(x)
>>> y2 = np.cos(x)
>>> plt.plot(x, y1, label="sin")
[<matplotlib.lines.Line2D object at 0x10d3df0d0>]
>>> plt.plot(x, y2, linestyle = "--", label="cos") # 用虚线绘制
[<matplotlib.lines.Line2D object at 0x106239950>]
>>> plt.xlabel("x") # x轴标签
Text(0.5, 47.04444444444444, 'x')
>>> plt.ylabel("y") # y轴标签
Text(90.44444444444443, 0.5, 'y')
>>> plt.title('sin & cos') # 标题
Text(0.5, 1.0, 'sin & cos')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x1062d2510>
>>> plt.show()
```

## 图片显示
plt-fox.png
```python
>>> import matplotlib.pyplot as plt
>>> from matplotlib.image import imread
>>> img = imread('fox.jpg') # 读入图像（设定合适的路径！）
>>> plt.imshow(img)
<matplotlib.image.AxesImage object at 0x127e13b50>
>>> plt.show()
2025-05-11 12:34:58.528 Python[63105:5270537] +[IMKClient subclass]: chose IMKClient_Legacy
2025-05-11 12:34:58.528 Python[63105:5270537] +[IMKInputSession subclass]: chose IMKInputSession_Legacy
```