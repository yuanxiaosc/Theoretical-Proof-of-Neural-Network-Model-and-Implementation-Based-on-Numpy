'''
dynamic_programming_accelerate_RNN_back_propagation.py

加速循环神经网络梯度反向传播的用动态规划算法。

为什么要写此模块：
1. 为了深刻理解循环神经网络进而理解深度学习；
2. 神经网络框架源码读不懂，还是自己实现吧；

功能：加速循环神经网络梯度在循环层的计算速度。

说明：
1. 为了提高运算效率，前馈神经网络基于 numpy 库来做矩阵运算，很少使用 for 循环。
2. 因为 numpy 中的某些机制比如“广播”等等虽然可以带来方便，但是也能带来难以查询和调试的错误，
   所以我宁愿花更多的代码来控制手中的数据，使它们完全被我操控。所以部分代码是可以写的更简洁。
3. 毕竟我只是为了尽可能的展示原理，很少考虑代码如何更高效，所以欢迎读者来优化这份代码，尽请修改吧！

'''


# Third-party libraries
import numpy as np


class Dynamic_programming_accelerate_RNN_back_propagation(object):
    '''
    input (Delta_Psi_matrix,C_Z_activation_function_gradient_vector, memory_vector_weight_S_matrix)
    output C_Z_memory_L
    输入：（梯度距离矩阵，循环层中相邻 Z 两点间的梯度距离，记忆向量的权重矩阵）
    输出：（损失函数对循环层 Z 所有时间步的梯度）
    '''
    def __init__(self, Delta_Psi_matrix,C_Z_activation_function_gradient_vector, memory_vector_weight_S_matrix):
        # 时间步数，也就是 RNN 的循环次数
        self.time_step = Delta_Psi_matrix.shape[0]
        # Psi 梯度距离矩阵。存储来自循环层上一层的梯度。
        self.Delta_Psi_matrix = Delta_Psi_matrix
        # Omega 梯度距离矩阵。存储循环层任两点的梯度距离。
        self.Omega_gradient_distance_matrix = None
        # 记忆向量的权重矩阵。
        self.S_matrix = memory_vector_weight_S_matrix
        # RNN 循环层（记忆层）中相邻 Z 两点间的梯度距离。
        self.contiguous_two_points_Omega_distance = C_Z_activation_function_gradient_vector
        # 代价函数对循环层 Z 矩阵的梯度矩阵。是整个类的返回值。
        self.C_Z_gradient_matrix = None

    # 初始化 Omega 梯度距离矩阵
    def initialization_Omega_gradient_distance_matrix(self):
        # 零填充矩阵，用来填充不使用的 Omega 距离矩阵
        zero_filled_matrix = np.zeros_like(self.contiguous_two_points_Omega_distance[0][0])
        self.Omega_gradient_distance_matrix = np.empty((self.time_step, self.time_step), dtype=np.object)
        # 先用零填充矩阵填充动态规划梯度矩阵，填充矩阵对角线及以上的位置
        for j in range(self.time_step):
            for i in range(self.time_step):
                if j <= i:
                    self.Omega_gradient_distance_matrix[j][i] = zero_filled_matrix
        # 记忆层中相邻 Z 两点间的梯度距离初始化动态规划梯度矩阵
        for j in range(1, self.time_step):
            self.Omega_gradient_distance_matrix[j][j-1] = self.contiguous_two_points_Omega_distance[j-1][0]

    # 使用动态规划算法填充 Omega 梯度距离矩阵
    def fill_Omega_derivative_distance_matrix(self):
        # 根据初始化的值逐步填充动态规划梯度矩阵，虽然有三个循环，其实复杂度只有 time_step 的平方。
        for j_i_difference_value in range(2, self.time_step):
            for j in range(self.time_step):
                for i in range(self.time_step):
                    if (j - i) == j_i_difference_value:
                        self.Omega_gradient_distance_matrix[j][i] = self.Omega_gradient_distance_matrix[j][i + 1] * \
                                                                 self.Omega_gradient_distance_matrix[i + 1][i]

    # 记忆向量的权重矩阵各个次方乘以 Omega 梯度距离矩阵中对应元素
    def fast_s_matrix_powers_maxtrix(self):
        # 这里提前把 S_matrix 各个次方提前计算，并且放在 Omega 梯度距离矩阵中，可以大大减少时间复杂度。
        S_matrix_powers_maxtrix = np.empty((self.time_step, 1), dtype=np.object)
        S_matrix_powers_maxtrix[1][0] = self.S_matrix.T
        for power in range(2, self.time_step):
            S_matrix_powers_maxtrix[power][0] = S_matrix_powers_maxtrix[power - 1][0] * self.S_matrix.T
        for j_i_difference_value in range(1, self.time_step):
            for j in range(self.time_step):
                for i in range(self.time_step):
                    if (j - i) == j_i_difference_value:
                        self.Omega_gradient_distance_matrix[j][i] = np.dot(self.Omega_gradient_distance_matrix[j][i],S_matrix_powers_maxtrix[j_i_difference_value][0])


    # 再次使用动态规划算法，根据填充好的 Omega 梯度距离矩阵，计算循环层的梯度 C_Z
    def calculating_cyclic_layer_gradient(self):
        # 初始化 Omega 梯度距离矩阵
        self.initialization_Omega_gradient_distance_matrix()
        # 使用动态规划算法填充 Omega 梯度距离矩阵
        self.fill_Omega_derivative_distance_matrix()
        # 记忆向量的权重矩阵各个次方乘以 Omega 梯度距离矩阵中对应元素
        self.fast_s_matrix_powers_maxtrix()

        # 用0 矩阵初始化代价函数对循环层 Z 矩阵的梯度矩阵。
        self.C_Z_gradient_matrix = np.empty(self.Delta_Psi_matrix.shape, dtype=np.object)
        zero_filled_matrix = np.zeros_like(self.Delta_Psi_matrix[0][0])
        for j in range(self.Delta_Psi_matrix.shape[0]):
            self.C_Z_gradient_matrix[j][0] = zero_filled_matrix
        # 先计算最后一个时间步的 C_Z_T
        self.C_Z_gradient_matrix[-1][0] = self.Delta_Psi_matrix[-1][0]
        # 从倒数第二列到第一列遍历 Omega 动态规划梯度矩阵，也就是依次计算 C_Z_(T-1),C_Z_(T-2),,,C_Z(0)
        for i in range(self.time_step - 2, -1, -1):
            self.C_Z_gradient_matrix[i][0] = np.sum(self.Omega_gradient_distance_matrix[:, i].reshape(-1, 1) * \
                                   self.C_Z_gradient_matrix, axis=0, keepdims=True)[0][0] + self.Delta_Psi_matrix[i][0]
        # 特别注意
        # self.C_Z_gradient_matrix[i][0] = np.sum(self.Omega_gradient_distance_matrix[:, i].reshape(-1, 1) * \
        #                           self.C_Z_gradient_matrix, axis=0, keepdims=True)[0][0] + self.Delta_Psi_matrix[i][0]
        # 与 self.C_Z_gradient_matrix[i][0] = np.sum(self.Omega_gradient_distance_matrix[:, i].reshape(-1, 1) * \
        #                           self.C_Z_gradient_matrix) + self.Delta_Psi_matrix[i][0]
        # 效果一致，但是我采用了写法更复杂的前者
        return self.C_Z_gradient_matrix



    def show_Omega_gradient_distance_matrix(self):
        for j in range(self.time_step):
            for i in range(self.time_step):
                print('下面是第({}, {})元素:\n{}'.format(j,i,self.Omega_gradient_distance_matrix[j][i]))


