from RNN.dynamic_programming_accelerate_RNN_back_propagation import Dynamic_programming_accelerate_RNN_back_propagation
import numpy as np

# 构造数据
# sigmoid 激活函数
def sigmoid(F):
    return 1.0 / (1 + np.exp(-F))

# sigmoid 激活函数的导数
def sigmoid_gradient(F):
    return sigmoid(F) * (1 - sigmoid(F))

import random

time_step = 5
sample_feature_dimension = 3
batch_size = 7

k1 = np.array([1, 0, 0])
k2 = np.array([0, 1, 0])
k3 = np.array([0, 0, 1])

List_K = [k1, k2, k3]

X_data_list = []
for i in range(time_step*batch_size):
    xx = random.choice(List_K)
    X_data_list.append(xx)
X_data = np.array(X_data_list).reshape(time_step, batch_size, sample_feature_dimension)
Y_data = np.copy(X_data)

#print(X_data.shape)

d0 = 3
d1 = 4
d2 = 5
d3 = 4
d4 = 3

W1 = np.random.randn(d1, d0)
W2 = np.random.randn(d2, d1)
W3 = np.random.randn(d3, d2)
W4 = np.random.randn(d4, d3)
S3 = np.random.randn(d3, d3)
b1 = np.zeros((d1, 1))
b2 = np.zeros((d2, 1))
b3 = np.zeros((d3, 1))
b4 = np.zeros((d4, 1))

X_train, Y_train = X_data, Y_data
H3_t0 = np.zeros((batch_size, d3))
# 前向传播
A0_t0 = X_train[0]
A0_t1 = X_train[0]
A0_t2 = X_train[1]
A0_t3 = X_train[2]
A0_t4 = X_train[1]

Z1_t0 = np.dot(A0_t0, W1.T) + b1.T
Z1_t1 = np.dot(A0_t1, W1.T) + b1.T
Z1_t2 = np.dot(A0_t2, W1.T) + b1.T
Z1_t3 = np.dot(A0_t3, W1.T) + b1.T
Z1_t4 = np.dot(A0_t3, W1.T) + b1.T

A1_t0 = sigmoid(Z1_t0)
A1_t1 = sigmoid(Z1_t1)
A1_t2 = sigmoid(Z1_t2)
A1_t3 = sigmoid(Z1_t3)
A1_t4 = sigmoid(Z1_t4)

Z2_t0 = np.dot(A1_t0, W2.T) + b2.T
Z2_t1 = np.dot(A1_t1, W2.T) + b2.T
Z2_t2 = np.dot(A1_t2, W2.T) + b2.T
Z2_t3 = np.dot(A1_t3, W2.T) + b2.T
Z2_t4 = np.dot(A1_t4, W2.T) + b2.T

A2_t0 = sigmoid(Z2_t0)
A2_t1 = sigmoid(Z2_t1)
A2_t2 = sigmoid(Z2_t2)
A2_t3 = sigmoid(Z2_t3)
A2_t4 = sigmoid(Z2_t4)

memory_C_Z_activation_function_gradient_vector = np.empty((time_step, 1), dtype=np.object)

# 有记忆向量的第三层只能串行，因为它们有依赖关系
#--------------串行开始---------------
Z3_t0 = np.dot(A2_t0, W3.T) + np.dot(H3_t0, S3.T) + b3.T
A3_t0 = sigmoid(Z3_t0)
H3_t1 = A3_t0
#memory_C_Z_activation_function_gradient_vector[0][0] = sigmoid_gradient(Z3_t0)
memory_C_Z_activation_function_gradient_vector[0][0] = np.ones_like(Z3_t0)

Z3_t1 = np.dot(A2_t1, W3.T) + np.dot(H3_t1, S3.T) + b3.T
A3_t1 = sigmoid(Z3_t1)
H3_t2 = A3_t1
#memory_C_Z_activation_function_gradient_vector[1][0]  = sigmoid_gradient(Z3_t1)
memory_C_Z_activation_function_gradient_vector[1][0] = np.ones_like(Z3_t1)

Z3_t2 = np.dot(A2_t2, W3.T) + np.dot(H3_t2, S3.T) + b3.T
A3_t2 = sigmoid(Z3_t2)
H3_t3 = A3_t2
#memory_C_Z_activation_function_gradient_vector[2][0]  = sigmoid_gradient(Z3_t2)
memory_C_Z_activation_function_gradient_vector[2][0] = np.ones_like(Z3_t2)

Z3_t3 = np.dot(A2_t3, W3.T) + np.dot(H3_t3, S3.T) + b3.T
A3_t3 = sigmoid(Z3_t3)
H3_t4 = A3_t3
#memory_C_Z_activation_function_gradient_vector[3][0]  = sigmoid_gradient(Z3_t3)
memory_C_Z_activation_function_gradient_vector[3][0] = np.ones_like(Z3_t3)

Z3_t4 = np.dot(A2_t4, W3.T) + np.dot(H3_t4, S3.T) + b3.T
A3_t4 = sigmoid(Z3_t4)
H3_t5 = A3_t4
#memory_C_Z_activation_function_gradient_vector[4][0]  = sigmoid_gradient(Z3_t4)
memory_C_Z_activation_function_gradient_vector[4][0] = np.ones_like(Z3_t4)

#print(memory_C_Z_activation_function_gradient_vector)

memory_vector_weight_S_matrix = S3

# Psi 梯度距离矩阵
Delta_Psi_matrix = np.empty((time_step, 1), dtype=np.object)

# 1 填充矩阵
ones_filled_matrix = np.ones_like(Z3_t0)

for i in range(time_step):
    Delta_Psi_matrix[i][0] = ones_filled_matrix

memory_vector_weight_S_matrix = 0.5 * np.ones_like(memory_vector_weight_S_matrix)

DP_RNN_BP =  Dynamic_programming_accelerate_RNN_back_propagation(Delta_Psi_matrix,
                                                                 memory_C_Z_activation_function_gradient_vector,
                                                                 memory_vector_weight_S_matrix)

C_Z_memery_layer = DP_RNN_BP.calculating_cyclic_layer_gradient()



print("输入:\n")
print("Delta_Psi_matrix:\n")
#print(Delta_Psi_matrix.shape)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print("memory_C_Z_activation_function_gradient_vector:\n")
#print(memory_C_Z_activation_function_gradient_vector)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print("memory_vector_weight_S_matrix:\n")
#print(memory_vector_weight_S_matrix)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print('输出:\n')
#print("C_Z_memery_layer:\n")
#print(C_Z_memery_layer)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def c_Z_gradient_matrix_to_Delta_Psi_3_dims( C_Z_memery_matrix):
    temp_list = []
    for i in range(5):
        temp_list.append([C_Z_memery_matrix[i][0]])
    Delta_memery_layer = np.concatenate(temp_list, axis=0)
    return Delta_memery_layer

print(c_Z_gradient_matrix_to_Delta_Psi_3_dims(C_Z_memery_layer).shape)

