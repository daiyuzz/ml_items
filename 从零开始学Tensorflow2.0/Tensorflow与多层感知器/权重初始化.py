"""
神经网络及深度学习模型训练的本质是对权重进行更新，本节使用TensorFlow 2.0 对权重初始化进行介绍
"""

# 导入数据集
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 将(60000,28,28)转换为(60000,784)
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape(x_test.shape[0], -1)

# 导入MNIST 数据集
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# 构建一个添加权重值 kernel_initalizer='he_normal'的模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(784,)),
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

"""
在模型正确的前提下，对回归分析模型进行训练，本例中训练次数为100次，显示图像，代码如下
"""

# 制定训练计划，训练100次
history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)

# 绘图
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

result = model.evaluate(x_test, y_test)

"""
本例采用了与基本数据集相同的数据集，权重为he_normal。通过对比两者的图像he输出结果可以看出，
添加权重会对结果产生影响。选择合适的初始权重值对数据分析的结果和学习时间有很大的影响
"""

"""
说明：he_normal是He正态分布初始化方法，参数由均值为0，标准差为sqrt(2/fan_in)的正态分布产生的，其中fan_in为权重张量
"""
