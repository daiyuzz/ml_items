# -*- coding: utf-8 -*-
# @Time    : 2021/5/19 上午9:01
# @Author  : daiyu
# @File    : 4-5,AutoGraph和tf.Module.py
# @Software: PyCharm

import tensorflow as tf

x = tf.Variable(1.0, dtype=tf.float32)


# 在tf.fuction中应用input_signature限定输入张量的签名类型：shape和dtype
@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return x


add_print(tf.constant(3.0))


# 4
# #add_print(tf.constant(3)) #输入不符合张量签名的参数将报错

# 下面利用tf.Module的子类将其封装一下

class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:
            self.x = tf.Variable(init_value, dtype=tf.float32, trainable=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def addprint(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return self.x


demo = DemoModule(init_value=tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))
# 6

# 查看模块中全部变量和全部可训练变量
print(demo.variables)
print(demo.trainable_variables)
# (<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
# (<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)

# 查看模块中的全部子模块
print(demo.submodules)
