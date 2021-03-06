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

dy_dx = tape.gradient(y, x)
print(dy_dx)
# tf.Tensor(-2.0, shape=(), dtype=float32)


# 对常量张量也可以求导，需要增加watch

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_d, dy_dc = tape.gradient(y, [x, a, b, c])
print(dy_da)
# tf.Tensor(0.0, shape=(), dtype=float32)
print(dy_dc)
# tf.Tensor(1.0, shape=(), dtype=float32)

# 可以求二阶导数
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape1.gradient(y, x)
dy2_dx2 = tape2.gradient(dy_dx, x)

print(dy2_dx2)
# tf.Tensor(2.0, shape=(), dtype=float32)

# 可以在autograph中使用
@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    # 自变量转换成tf.float32
    x = tf.cast(x,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a*tf.pow(x,2)+b*x+c
    dy_dx = tape.gradient(y,x)

    return ((dy_dx,y))

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))
# (-2, 1)
# (0, 0)

## 二、利用梯度磁带和优化器求最小值
"""
求 f(x) = a*x**2+b*x +c的最小值
使用optimizer.apply_gradients 
"""
x = tf.Variable(0.0,name="x",dtype=tf.float32)
a = tf.constant(1.0)
b=tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2)+b*x+c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
tf.print("y = ",y,";x=",x)
# y =  0 ;x= 0.999998569



# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.minimize
# optimizer.minimize相当于先用tape求gradient,再apply_gradient

x = tf.Variable(0.0,name="x",dtype=tf.float32)

# 注意f()无参数
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2) + b*x + c
    return y

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    optimizer.minimize(f,[x])

tf.print("y=",f(),";x=",x)
# y= 0 ;x= 0.999998569



# 在autograph中完成最小值求解
# 使用optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    for _ in tf.range(1000):
        with tf.GradientTape() as tape:
            y = a*tf.pow(x,2)+b*x+c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])

    y = a*tf.pow(x,2) + b*x + c
    return y

tf.print(minimizef())
tf.print(x)

# 0
# 0.999998569


# 在autograph中完成最小值求解
# 使用optimizer.minimize
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f,[x])
    return(f())


tf.print(train(1000))
tf.print(x)

# 0
# 0.999998569