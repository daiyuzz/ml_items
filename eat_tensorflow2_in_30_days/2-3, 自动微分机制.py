# -*- coding: utf-8 -*-
# @Time    : 2021/5/17 上午8:31
# @Author  : daiyu
# @File    : 2-3, 自动微分机制.py
# @Software: PyCharm


# 利用梯度磁带求导数

import tensorflow as tf
import numpy as np

# f(x) = a*x**2 + b*x + c的导数

x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y,x)
print(dy_dx)
# tf.Tensor(-2.0, shape=(), dtype=float32)

