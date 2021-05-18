# -*- coding: utf-8 -*-
# @Time    : 2021/5/17 上午8:39
# @Author  : daiyu
# @File    : 4-1,张量的结构操作.py
# @Software: PyCharm

# 创建张量
import tensorflow as tf
import numpy as np

a = tf.constant([1,2,3],dtype=tf.float32)
tf.print(a)
# [1 2 3]

b = tf.range(1,10,delta=2)
tf.print(b)

c = tf.linspace(0.0,2*3.14,100)
tf.print(c)
# [0 0.0634343475 0.126868695 ... 6.15313148 6.21656609 6.28]

d = tf.zeros([3,3])
tf.print(d)
# [[0 0 0]
#  [0 0 0]
#  [0 0 0]]

a = tf.ones([3,3])
b = tf.zeros_like(a,dtype=tf.float32)
tf.print(a)
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]]
tf.print(b)
# [[0 0 0]
#  [0 0 0]
#  [0 0 0]]

b = tf.fill([3,2],5)
tf.print(b)
# [[5 5]
#  [5 5]
#  [5 5]]

# 均匀分布随机
tf.random.set_seed(1.0)

a = tf.random.uniform([5],minval=0,maxval=10)
tf.print(a)
# [1.65130854 9.01481247 6.30974197 4.34546089 2.9193902]

# 正态随机分布
b = tf.random.normal([3,3],mean=0.0,stddev=1.0)
tf.print(b)
# [[0.403087884 -1.0880208 -0.0630953535]
#  [1.33655667 0.711760104 -0.489286453]
#  [-0.764221311 -1.03724861 -1.25193381]]

# 正太随机分布，剔除2倍方差以外数据重新生成
c = tf.random.truncated_normal((5,5),mean=0.0,stddev=1.0,dtype=tf.float32)
tf.print(c)
# [[-0.457012236 -0.406867266 0.728577733 -0.892977774 -0.369404584]
#  [0.323488563 1.19383323 0.888299048 1.25985599 -1.95951891]
#  [-0.202244401 0.294496894 -0.468728036 1.29494202 1.48142183]
#  [0.0810953453 1.63843894 0.556645 0.977199793 -1.17777884]
#  [1.67368948 0.0647980496 -0.705142677 -0.281972528 0.126546144]]


# 特殊矩阵
I = tf.eye(3,3) #单位矩阵
tf.print(I)
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]

tf.print(" ")
t = tf.linalg.diag([1,2,3]) #对角矩阵
tf.print(t)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]



# 二、索引切片

tf.random.set_seed(3)
t = tf.random.uniform([5,5],minval=0,maxval=10,dtype=tf.int32)
tf.print(t)

# [[4 7 4 2 9]
#  [9 1 2 4 7]
#  [7 2 7 4 0]
#  [9 6 9 7 2]
#  [3 7 0 0 3]]

# 第0行
tf.print(t[0])
# [4 7 4 2 9]

# 倒数第一行
tf.print(t[-1])
# [3 7 0 0 3]

# 第1行第3列
tf.print(t[1,3])
# 4
tf.print(t[1][3])
# 4

# 第一行至第三行
tf.print(t[1:4,:])
tf.print(tf.slice(t,[1,0],[3,5])) # tf.slice(input,begin_vector,size_vector)
# [[9 1 2 4 7]
#  [7 2 7 4 0]
#  [9 6 9 7 2]]

# 第一行至最后一行，第0列到最后一列每隔两列取一列
tf.print(t[1:4,:4:2])
# [[9 2]
#  [7 7]
#  [9 9]]

# 对变量来说，还可以使用索引和切片修改部分元素
x = tf.Variable([[1,2],[3,4]],dtype=tf.float32)
x[1,:].assign(tf.constant([0.0,0.0]))
tf.print(x)
# [[1 2]
#  [0 0]]

a = tf.random.uniform([3,3,3],minval=0,maxval=10,dtype=tf.int32)
tf.print(a)
# [[[7 3 9]
#   [9 0 7]
#   [9 6 7]]
#  [[1 3 3]
#   [0 8 1]
#   [3 1 0]]
#  [[4 0 6]
#   [6 2 2]
#   [7 9 5]]]

tf.print(a[...,1])
# [[3 0 6]
#  [3 8 1]
#  [0 2 9]]

