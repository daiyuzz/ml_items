"""
本节构建一个基础模型，以与后面章节中的优化模型进行对比
"""

# 导入数据集
# 以tensorflow为基础构建keras 并导入数据集
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 将(60000,28,28)转换为(60000,784)
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape(x_test.shape[0], -1)

# 导入MNIST 数据集
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# 构建一个不添加任何条件的模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
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
plt.legend(['training','validation'],loc='upper left')
plt.show()
