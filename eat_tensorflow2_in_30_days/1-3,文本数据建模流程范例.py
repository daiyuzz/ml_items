import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing, optimizers,losses,metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string

train_data_path = "./data/imdb/train.csv"
test_data_path = "./data/imdb/test.csv"

MAX_WORDS = 10000  # 仅考虑最高词频的10000个词
MAX_LEN = 200
BATCH_SIZE = 20


# 构建管道
def split_line(line):
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)  # tf.cast 数据格式转换
    text = tf.expand_dims(arr[1], axis=0)
    return (text, label)


ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


# 构建词典
def clean_text(text):
    lowercase = tf.strings.lower(text)  # 将文本大写字母转换成小写字母
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')  # tf.strings.regex_replace 将html里面的标签转化为空格
    cleaned_punctuation = tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation),
                                                   '')  # 去除所有标点符号
    return cleaned_punctuation


# 构建向量化层
vectorize_layer = TextVectorization(
    standardize=clean_text,
    split='whitespace',
    max_tokens=MAX_WORDS - 1,  # 有一个留给占位符
    output_mode='int',
    output_sequence_length=MAX_LEN
)

ds_text = ds_train_raw.map(lambda text, label: text)  # 取ds_train_raw里面的text
vectorize_layer.adapt(ds_text)
print(vectorize_layer.get_vocabulary()[0:100])

# 单词编码(向量化)
ds_train = ds_train_raw.map(lambda text,label:(vectorize_layer(text),label)).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text,label:(vectorize_layer(text),label)).prefetch(tf.data.experimental.AUTOTUNE)

# ds_text = ds_train_raw.map(lambda text,label: text)这里之所以要这么写，是因为ds_train_raw里不仅包含了label还包含了text,但是这里只需要text就够了，因此通过匿名函数的方式将返回的text和label只取text。这里的text和label是之前ds_train_raw调用之前的splitline函数所返回的结果



# 定义模型
tf.keras.backend.clear_session()

class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self,input_shape):
        self.embedding = layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16,kernel_size=5,name="conv_1",activation="relu")
        self.pool_1 = layers.MaxPool1D(name="pool_1")
        self.conv_2 = layers.Conv1D(128,kernel_size=2,name="conv_2",activation="relu")
        self.pool_2 = layers.MaxPool1D(name="pool_2")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1,activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self,x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    #用于显示 Output Shape
    def summary(self):
        x_input = layers.Input(shape=MAX_LEN)
        output = self.call(x_input)
        model = tf.keras.Model(inputs=x_input,outputs=output)
        model.summary()


model = CnnModel()
model.build(input_shape=(None,MAX_LEN))
model.summary()



# 训练模型

# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)


optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):

        for features, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)

        # 此处logs模板需要根据metric具体情况修改
        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                        valid_metric.result())))
            tf.print("")

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


train_model(model, ds_train, ds_test, epochs=6)
