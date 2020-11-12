"""
在数据规模较大的情况下，单个模型的效率和预测结果并不能令人满意。因此，Tensorflow2.0提供了模型集成算法，
可以将多个模型组合使用，能够得到更好的预测结果
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])


# 制定一个dropout为0.2的基础模型
def mlp_model():
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
    return model


# 使用KerasClassifier，根据基础模型构建新的模型
model1 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model2 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model3 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)

# 算法为voting='soft'
ensemble_clf = VotingClassifier(estimators=[('model1', model1), ('model2', 'model2'), ('model3', 'model3')],
                                voting='soft')

ensemble_clf.fit(x_train, y_train)
y_pred = ensemble_clf.predict(x_test)
print("acc", accuracy_score(y_pred, y_test))
