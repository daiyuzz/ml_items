"""
dropout(随机失活)是对具有深度结构的人工神经网络进行优化的方法，在学习过程中，通过将隐含层的部分权重
或输出随机归零来降低节点之间的依赖性、实现神经网络的正则化、降低其结构风险并实现优化目标
"""

import ssl
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.SGD(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=256,epochs=100,validation_split=0.3,verbose=0)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'],loc='upper left')
plt.show()
result = model.evaluate(x_test,y_test)
print(result)


"""
本例采用了与基本数据集相同的数据集，并令每个层的rate为0.2，以防止过拟合。其意义在于按比例1/(1-rate)对层
进行缩放，以使他们在训练和推理时间内的总和不变。
"""