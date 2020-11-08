from tensorflow import keras
from tensorflow.keras import layers

# 波士顿房价数据
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# 以boston_housing数据为基础构建模型

# 用sigmoid算法构建一个线性时序模型
model = keras.Sequential([
    layers.Dense(32, activation='sigmoid', input_shape=(13,)),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(1)
])

# 使用密集连接层类构建模型
model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss='mean_squared_error',
              metrics=['mse'])

model.summary()

# 迭代训练模型
model.fit(x_train, y_train, batch_size=50, epochs=50, validation_split=0.1, verbose=1)

result = model.evaluate(x_test, y_test)

print(model.metrics_names)
print(result)
