"""
批标准化要解决的问题是：模型参数在学习阶段的变化使每个隐藏层输出的分布发生变化。这意味着靠后的层
要在训练过程中适应这些变化。批标准化是一种简单、高效的改善神经网络性能的方法
"""

# 构建批标准化模型，代码如下

import ssl
from tensorflow import keras
from tensorflow.keras import layers
## 导入图形化工具
import matplotlib.pyplot as plt

## 由于是HTTPS协议域名，需要添加相应证书
ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])

## 对每个层进行标准化，具体参数为layers.BatchNormalization()
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

## 编译模型
model.compile(optimizer=keras.optimizers.SGD(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

## 制定训练计划，训练100次
history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)

# 打印输出结果
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(['training', 'validation'], loc='upper left')
plt.show()
print(history.history)

result = model.evaluate(x_test, y_test)
print(result)


"""
本例采用了与基本数据集相同的数据集，并对每个层进行了批标准化。通过对比两者的图像输出结果可以看出，
在批标准化后，结果的精度和训练速度均有改善
"""