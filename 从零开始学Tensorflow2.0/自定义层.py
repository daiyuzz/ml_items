import tensorflow as tf

# 指定网络的神经元个数
layer = tf.keras.layers.Dense(100)

# 添加输入维度限制
layer = tf.keras.layers.Dense(100, input_shape=(None, 20))

# 每层都可以作为一个函数，将输入的数据作为函数的输入
layer(tf.ones([6, 6]))

# 分别取出权重和偏置
print(layer.kernel, layer.bias)

# 自定义网络层
