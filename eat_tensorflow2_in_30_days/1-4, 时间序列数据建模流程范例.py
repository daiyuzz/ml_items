# -*- coding: utf-8 -*-
# @Time    : 2021/5/14 上午8:40
# @Author  : daiyu
# @File    : 1-4, 时间序列数据建模流程范例.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, callbacks

df = pd.read_csv('./data/covid-19.csv', sep="\t")
df.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10, 6))
plt.xticks(rotation=60)
# plt.show()

dfdata = df.set_index("date")
dfdiff = dfdata.diff(periods=1).dropna()  # 用于查找同一系列元素之间的差异,periods=1,即当前元素和前一个比较;dropna()是移除NA

dfdiff = dfdiff.reset_index("date")

dfdiff.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10, 6))
plt.xticks(rotation=60)  # 横坐标文字倾斜
# plt.show()


# 用某日前8天窗口数据作为输入预测该日的数据

WINDOW_SIZE = 8


def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched


ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values, dtype=tf.float32)) \
    .window(WINDOW_SIZE, shift=1).flat_map(batch_dataset)
