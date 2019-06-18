import numpy as np
# 设置固定的随机数，便于复现同样的模型输出
np.random.seed(1994)

d0 = 3
d1 = 4
d2 = 5
d3 = 4
d4 = 3

contain_h = False

time_step = 3
sample_feature_dimension = 3
batch_size = 10

X = np.random.random(size=(time_step, batch_size, sample_feature_dimension))
print(X.shape)
