# 从Numpy array 构建数据管道

import tensorflow as tf
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

ds1 = tf.data.Dataset.from_tensor_slices((iris["data"], iris["target"]))

for features, label in ds1.take(5):
    print(features, label)

# 从pands DataFrame构建数据管道
import tensorflow as tf
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
dfiris = pd.DataFrame(iris["data"], columns=iris.feature_names)
ds2 = tf.data.Dataset.from_tensor_slices(
    (dfiris.to_dict("list"), iris["target"]))  # ‘list’ : dict like {column -> [values]}

for features, label in ds2.take(3):
    print(features, label)

# {'sepal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=5.1>, 'sepal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=3.5>, 'petal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=1.4>, 'petal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=0.2>} tf.Tensor(0, shape=(), dtype=int64)
# {'sepal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=4.9>, 'sepal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=3.0>, 'petal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=1.4>, 'petal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=0.2>} tf.Tensor(0, shape=(), dtype=int64)
# {'sepal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=4.7>, 'sepal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=3.2>, 'petal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=1.3>, 'petal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=0.2>} tf.Tensor(0, shape=(), dtype=int64)


# 从python generator构建数据管道

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义一个从文件中读取图片的generator
image_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    "data/cifar2/test",
    target_size=(32, 32),
    batch_size=20,
    class_mode='binary'
)

classdict = image_generator.class_indices
print(classdict)


def generator():
    for features, label in image_generator:
        yield (features, label)


ds3 = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))

plt.figure(figsize=(6, 6))
for i, (img, label) in enumerate(ds3.unbatch().take(9)):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# 从csv构建数据管道

# 从csv文件构建数据管道
ds4 = tf.data.experimental.make_csv_dataset(
    file_pattern=["./data/titanic/train.csv", "./data/titanic/test.csv"],
    batch_size=3,
    label_name="Survived",
    na_value="",
    num_epochs=1,
    ignore_errors=True
)

for data, label in ds4.take(2):
    print(data, label)

# 从文本文件构建数据管道

ds5 = tf.data.TextLineDataset(
    filenames=["./data/titanic/train.csv", "./data/titanic/test.csv"]
).skip(1)  # 略去第一行header

for line in ds5.take(5):
    print(line)

# 从文件路径构建数据管道
ds6 = tf.data.Dataset.list_files("./data/cifar2/train/*/*.jpg")
for file in ds6.take(5):
    print(file)

from matplotlib import pyplot as plt


def load_image(img_path, size=(32, 32)):
    label = 1 if tf.strings.regex_full_match(img_path, ".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
    img = tf.image.resize(img, size)
    return (img, label)


for i, (img, label) in enumerate(ds6.map(load_image).take(2)):
    plt.figure(i)
    plt.imshow((img / 255.0).numpy())
    plt.title("label = %d" % label)
    plt.xticks([])
    plt.yticks([])

# 从tfrecords文件构建数据管道
import os
import numpy as np


# inpath:原始数据路径  outpath:TFRecord文件输出路径
def create_tfrecords(inpath, outpath):
    writer = tf.io.TFRecordWriter(outpath)


## 应用数据转换

ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello beijing"])
ds_map = ds.map(lambda x: tf.strings.split(x, " "))
for x in ds_map:
    print(x)

# tf.Tensor([b'hello' b'world'], shape=(2,), dtype=string)
# tf.Tensor([b'hello' b'china'], shape=(2,), dtype=string)
# tf.Tensor([b'hello' b'beijing'], shape=(2,), dtype=string)

ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello beijing"])
ds_flatmap = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, ' ')))
for x in ds_flatmap:
    print(x)

# tf.Tensor(b'hello', shape=(), dtype=string)
# tf.Tensor(b'world', shape=(), dtype=string)
# tf.Tensor(b'hello', shape=(), dtype=string)
# tf.Tensor(b'china', shape=(), dtype=string)
# tf.Tensor(b'hello', shape=(), dtype=string)
# tf.Tensor(b'beijing', shape=(), dtype=string)


# interleave：效果类似 flat_map,但可以将不同来源的数据夹在一起

ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello Beijing"])
ds_interleave = ds.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for x in ds_interleave:
    print(x)
# tf.Tensor(b'hello', shape=(), dtype=string)
# tf.Tensor(b'hello', shape=(), dtype=string)
# tf.Tensor(b'hello', shape=(), dtype=string)
# tf.Tensor(b'world', shape=(), dtype=string)
# tf.Tensor(b'china', shape=(), dtype=string)
# tf.Tensor(b'Beijing', shape=(), dtype=string)


# filter 过滤掉某些元素
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello china","hello Beijing"])
# 找出含有字母a或字母B的元素
ds_filter = ds.filter(lambda x:tf.strings.regex_full_match(x,".*[a|B].*"))
for x in ds_filter:
    print(x)
# tf.Tensor(b'hello china', shape=(), dtype=string)
# tf.Tensor(b'hello Beijing', shape=(), dtype=string)



