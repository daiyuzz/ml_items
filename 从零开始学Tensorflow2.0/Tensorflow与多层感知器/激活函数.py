"""
激活函数在运行时，激活神经网络中的某部分神经元，并将激活神经元的
信息输入到下一层神经网络中。神经网络之所以能够处理非线性问题，是
因为激活函数具有非线性表达能力
"""

# 本节以sigmoid为例进行讲解，构建以sigmoid为激活函数的模型

import ssl
from tensorflow import keras
from tensorflow.keras import layers

# 由于是 HTTPS 协议域名，需要添加相应证书
ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])

# 构建一个添加激活函数activation='sigmoid'的模型
model = keras.Sequential([
    layers.Dense(64, activation='sigmoid', input_shape=(784,)),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)

# 绘图
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

result = model.evaluate(x_test, y_test)
print(result)

"""
sigmoid函数也称为logistic函数，用于隐层神经元输出，取值范围（0，1），
它可以将一个实数映射到（0，1）区间，可以进行二分类
"""

"""
本例采用了与基本数据集相同的数据集，定义sigmoid为激活函数。通过对比两者之间的图像和输出结果可以看出
指定激活函数对结果精确度的影响比较明显。说明sigmoid激活函数适合用于分析这个数据集
"""
