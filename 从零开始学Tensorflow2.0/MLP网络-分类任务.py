from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras

whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

# 以breast_cancer数据集为基础构建模型
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(30,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 使用密集连接层构建模型
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test)
print(model.metrics_names)
