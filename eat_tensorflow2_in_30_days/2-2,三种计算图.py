# -*- coding: utf-8 -*-
# @Time    : 2021/5/16 下午12:33
# @Author  : daiyu
# @File    : 2-2,三种计算图.py
# @Software: PyCharm

"""

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph

在 tensorflow1.0 时代，采用的是静态计算图，需要先使用Tensorflow的各种算子创建计算图，然后再开启一个会话Session，显式执行计算图

而在Tensorflow2.0 时代，采用的是动态计算图，即每使用一个算子后，该算子会被动态加入到隐含的默认计算图中立即执行得到结果，而无需开启Session

"""
from pathlib import Path

import tensorflow as tf

# tensorflow 1.0
# 定义计算图
# g = tf.Graph()
# with g.as_default():
#     # placeholder 为占位符，执行会话的时候执行填充对象
#     x = tf.placeholder(name='x',shape=[],dtype=tf.string)
#     y = tf.placeholder(name='y',shape=[],dtype=tf.string)
#     z = tf.string_join([x,y],name='join',separator='')
#
# # 执行计算图
# with tf.Session(graph=g) as sess:
#     print(sess.run(fetches=z,feed_dict={x:"hello",y:"world"}))

# tensorflow2.0 怀旧版静态计算图

import tensorflow as tf

g = tf.compat.v1.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(name='x', shape=[], dtype=tf.string)
    y = tf.compat.v1.placeholder(name='y', shape=[], dtype=tf.string)
    z = tf.strings.join([x, y], name="join", separator=" ")

with tf.compat.v1.Session(graph=g) as sess:
    # # fetches的结果非常像一个函数的返回值，而feed_dict中的占位符相当于函数的参数序列。
    result = sess.run(fetches=z, feed_dict={x: "hello", y: "world"})
    print(result)

# 动态计算图在每个算子处都进行构建，构建后立即执行

x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x, y], separator=" ")
tf.print(z)


# hello world

# 可以将动态计算图代码的输入和输出关系封装成函数
def strjoin(x, y):
    z = tf.strings.join([x, y], separator=" ")
    tf.print(z)
    return z


result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)

import tensorflow as tf


# 使用autograph构建静态图

@tf.function
def strjoin(x, y):
    z = tf.strings.join([x, y], separator=" ")
    tf.print(z)
    return z


result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)

# hello world
# tf.Tensor(b'hello world', shape=(), dtype=string)


import datetime
from pathlib import Path
# 创建日志
import os

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path('./data/autograph' + stamp))

writer = tf.summary.create_file_writer(logdir)

# 开启autograph跟踪
tf.summary.trace_on(graph=True, profiler=True)

# 执行autograph
result = strjoin("hello", "world")

# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name="autograph",
        step=0,
        profiler_outdir=logdir
    )

# 启动 tensorboard在jupyter中的魔法命令
# %load_ext tensorboard

# 启动tensorboard
# %tensorboard --logdir ./data/autograph/
