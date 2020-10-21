import tensorflow as tf

# 从列表中获取Tensor
ds_tensors = tf.data.Dataset.from_tensor_slices([6, 5, 4, 3, 2, 1])

# 创建 CSV 文件
import tempfile

# tempfile 模块专门用于创建临时文件和临时目录

_, filename = tempfile.mkstemp()
print(filename)

# 循环打开文件，获得内容
with open(filename, 'w') as f:
    f.write("""Line 1 
    Line 2 
    Line 3""")

# 获取TextLineDataset 数据集实例
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# shuffle(2) 将数据打乱，数值越大，混乱程度越大
# batch(2) 按照顺序取出2行数据，最后一次输出可能小于batch,也就是其shape=(2,)

for x in ds_tensors:
    print(x)

for x in ds_file:
    print(x)
