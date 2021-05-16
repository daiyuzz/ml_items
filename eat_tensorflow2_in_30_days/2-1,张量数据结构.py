# -*- coding: utf-8 -*-
# @Time    : 2021/5/16 上午10:32
# @Author  : daiyu
# @File    : 2-1,张量数据结构.py
# @Software: PyCharm

"""
程序 = 数据结构 + 算法
Tensorflow程序 = 张量数据结构 + 计算图算法语言
张量和计算图是 tensorflow的核心概念
Tensorflow的基本数据结构是张量Tensor。张量即多维数组。Tensorflow的张量和numpy中的array很类似
从行为特性来看，有两种类型的张量，常量constant和变量Variable
常量的值在计算图中不可以被重新赋值，变量可以在计算图中用assign等算子重新赋值
"""

# 一、常量张量

# 张量的数据类型和numpy.array基本一一对应
import numpy as np
import tensorflow as tf

i = tf.constant(1)  # tf.int32 类型常量
l = tf.constant(1, dtype=tf.int64)  # tf.int64 类型常量
f = tf.constant(1.23)  # tf.float32类型常量
d = tf.constant(3.14, dtype=tf.double)  # tf.double 类型常量
s = tf.constant("hello world")  # tf.string 类型常量
b = tf.constant(True)  # tf.bool类型常量

print(tf.int64 == np.int64)
# True

print(tf.bool == np.bool)
# True

print(tf.double == np.float64)
# True

print(tf.string == np.unicode)
# False

"""
不同类型的数据可以用不同维度（rank）的张量来表示
标量为0维张量，向量为1维张量，矩阵为2维张量
视频还有时间维，可以表示为4维张量
可以简单的总结为：有几层括号，就是多少维张量
"""

scalar = tf.constant(True)  # 标量，0维张量
print(tf.rank(scalar))
# tf.Tensor(0, shape=(), dtype=int32)
print(scalar.numpy().ndim)  # tf.rank 的作用和numpy的ndim方法相同
# 0

vector = tf.constant([1.0, 2.0, 3.0, 4.0])  # 向量，1维张量
print(tf.rank(vector))
# tf.Tensor(1, shape=(), dtype=int32)
print(np.ndim(vector.numpy()))  # tf.rank 和 np.dim都是返回维度
# 1


matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # 矩阵，2维张量
print(tf.rank(matrix).numpy())
# 2
print(np.ndim(matrix))
# 2

tensor3 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # 3维向量
print(tensor3)
# tf.Tensor(
# [[[1. 2.]
#   [3. 4.]]
#
#  [[5. 6.]
#   [7. 8.]]], shape=(2, 2, 2), dtype=float32)
print(tf.rank(tensor3))
# tf.Tensor(3, shape=(), dtype=int32)

tensor4 = tf.constant([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
                       [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]]])  # 4维张量
print(tensor4)
# tf.Tensor(
# [[[[1. 1.]
#    [2. 2.]]
#
#   [[3. 3.]
#    [4. 4.]]]
#
#
#  [[[5. 5.]
#    [6. 6.]]
#
#   [[7. 7.]
#    [8. 8.]]]], shape=(2, 2, 2, 2), dtype=float32)
print(tf.rank(tensor4))
# tf.Tensor(4, shape=(), dtype=int32)

"""
tf.cast 改变张量的数据类型
可以用numpy方法将tensorflow中的张量转化为numpy中的张量
可以用shape方法查看张量的尺寸
"""

h = tf.constant([123, 456], dtype=tf.int32)
f = tf.cast(h, tf.float32)
print(h.dtype, f.dtype)
# <dtype: 'int32'> <dtype: 'float32'>

y = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(y)
# tf.Tensor(
# [[1. 2.]
#  [3. 4.]], shape=(2, 2), dtype=float32)
print(y.numpy())
# [[1. 2.]
#  [3. 4.]]
print(y.shape)
# (2, 2)


u = tf.constant(u"你好，世界")
print(u.numpy())
# b'\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c'
print(u.numpy().decode("utf-8"))
# 你好，世界

# 二、变量张量
# 模型中需要被训练的参数一般被设置为变量

# 常量值不可以改变，常量的重新赋值相当于创建新的内存空间
c = tf.constant([1.0, 2.0])
print(c)
# tf.Tensor([1. 2.], shape=(2,), dtype=float32)
print(id(c))
# 139649975359456
c = c + tf.constant([1.0, 1.0])
print(c)
# tf.Tensor([2. 3.], shape=(2,), dtype=float32)
print(id(c))
# 139649975359984


# 变量的值可以改变，可以通过assign，assign_add等方法给变量重新赋值
v = tf.Variable([1.0, 2.0], name="v")
print(v)
# <tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
print(id(v))
# 140217343239440
v.assign_add([1.0, 1.0])
print(v)
# <tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>
print(id(v))
# 140217343239440

