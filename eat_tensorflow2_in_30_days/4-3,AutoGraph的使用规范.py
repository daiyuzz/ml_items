import numpy as np
import tensorflow as tf

@tf.function
def np_random():
    a = np.random.randn(3,3)
    tf.print(a)

@tf.function
def tf_random():
    a = tf.random.normal((3,3))
    tf.print(a)

# np.random每次执行都是一样的结果
np_random()
np_random()

# array([[-2.1792799 , -1.45075419,  1.53125576],
#        [-1.86772072, -0.17847839, -0.49542453],
#        [-0.09012711, -1.19993147,  0.42333767]])
# array([[-2.1792799 , -1.45075419,  1.53125576],
#        [-1.86772072, -0.17847839, -0.49542453],
#        [-0.09012711, -1.19993147,  0.42333767]])

# tf.random每次执行都会有重新生成随机数
tf_random()
tf_random()
# [[-0.00578217534 0.928719044 -0.098365]
#  [1.38043475 0.272096336 -1.76296246]
#  [-1.05469251 0.0894044861 -0.429012328]]
# [[1.47495162 -0.8452667 -0.363954872]
#  [-0.923432529 -0.157690659 0.528821]



# 避免在@tf.function修饰的函数内部定义tf.Variable

x = tf.Variable(1.0, dtype=tf.float32)
@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return x

outer_var()
# 2
outer_var()
# 3


@tf.function
def inner_var():
    x = tf.Variable(1.0,dtype=tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return x

# 执行报错
# inner_var()
# inner_var()



# **3,被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等结构类型变量。**

tensor_list = []
@tf.function
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
#预期结果为：
#[<tf.Tensor: shape=(), dtype=float32, numpy=5.0>, <tf.Tensor: shape=(), dtype=float32, numpy=6.0>]
# 实际结果不符合预期
# [<tf.Tensor 'x:0' shape=() dtype=float32>]