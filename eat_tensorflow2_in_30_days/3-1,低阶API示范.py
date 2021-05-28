# -*- coding: utf-8 -*-
# @Time    : 2021/5/28 上午8:53
# @Author  : daiyu
# @File    : 3-1,低阶API示范.py
# @Software: PyCharm

# 下面的范例使用Tensorflow的低阶API实现线性回归模型和DNN二分类模型
# 低阶API主要包括张量操作，计算图和自动微分

import tensorflow as tf


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


"""
# 线性回归模型
"""

# 数据准备

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

# 样本数量
n = 400

# 生成测试用数据集
X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)  # @表示矩阵乘法，增加正态扰动

# 数据可视化
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121)  # 子图1行2列第一个图
ax1.scatter(X[:, 0], Y[:, 0], c="b")
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)  # 子图1行2列
ax2.scatter(X[:, 1], Y[:, 0], c="g")
plt.xlabel("x2")
plt.ylabel("y", rotation=0)

plt.show()


# 构建数据管道

def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        indexs = indices[i:min(i + batch_size, num_examples)]
        yield tf.gather(features, indexs), tf.gather(labels, indexs)

# 测试数据管道效果
batch_size = 8
(features,labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)
