from RNN.dynamic_programming_accelerate_RNN_back_propagation import Dynamic_programming_accelerate_RNN_back_propagation

# Third-party libraries
import numpy as np

# Python 的内置函数
from functools import reduce

# 基于矩阵运算的循环神经网络
class Recurrent_neural_network(object):

    def __init__(self):
        # T_Layers_data 存储网络每个时间步的数据，数据结构是[ [{},{},{}], [{},{},{}], [{},{},{}]]
        # 内容是 [ [时间步0] , [{第0层},{'Z':matrix, 'A':matrix},···,{第L层}], ··· ,[时间步T]  ]
        # 获取某个数据 'H' 方法是，T_Layers_data[时间步数][网络层数]['H']，数据维度是(1,样本个数，样本特征维度）
        self.T_Layers_data = []
        # Layers_weigh 存储网络权重，数据结构是 [{},{},{}]
        # 内容是 [{第0层},{'W':matrix,'b':matrix},{第L层},]
        # 获取某个数据 'W' 方法是，Layers_weight[网络层数]['W']，数据维度是(本层样本特征维度，上一层样本特征维度)
        self.Layers_weight = []
        # Layers_dims 列表是网络每一层的神经元个数
        self.Layers_dims = None
        # 网络层数
        self.layers_number = None
        # 时间步数
        self.time_steps_number = None
        # 记忆向量所在的层数
        self.memory_vector_layer_location = None
        # 小批量样本的批量数目
        self.batch_size = None
        # 损失函数名字
        self.loss_faction_name = None
        # ---------------------函数名与函数对象映射字典------------------------
        # 激活函数映射字典
        self.activation_functions_dict = {'sigmoid': self.sigmoid, 'softmax': self.softmax,}
        # 激活函数的导数函数的映射字典
        self.activation_gradient_functions_dict = {'sigmoid': self.sigmoid_gradient, 'relu': self.relu_gradient,
                                                   'leaky_relu': self.leaky_relu_gradient, 'softmax': None,
                                                   'None': self.equal_gradient}

    # ----------------------初始化权重网络结构------------------------
    def creating_network_weight_infrastructure(self, sample_feature_dimension,layer_Dims,
                                               activation_function_names, time_steps_number=None,
                                               memory_vector_layer_location=None):
        # 把输入层作为第 0 层，同时把它的维度插入维度数组中
        layer_Dims.insert(0, sample_feature_dimension)
        self.Layers_dims = layer_Dims
        #  默认含有记忆向量权重矩阵的层数是在倒数第二层，也就是输出层的前一层
        self.memory_vector_layer_location = len(self.Layers_dims) - 2
        # 初始化整个神经网络的权重，整个网络公用一套权重矩阵
        self.layers_number = len(self.Layers_dims)
        for layer_number in range(self.layers_number):
            layer = {}
            if layer_number == 0:
                self.Layers_weight.append(layer)
                continue
            layer['activation_function_name'] = activation_function_names[layer_number - 1]
            # 因为 relu 函数会把不大于 0 的值 设置为 0，导致模型在开始时候较难训练
            # 所以当某一层使用 relu 作为激活函数时，该层权重矩阵使用大于 0 值的初始化方法
            if layer['activation_function_name'] == 'relu' or layer['activation_function_name'] == 'leaky_relu':
                layer['W'] = np.random.uniform(np.exp(-10), 1,
                                               size=(layer_Dims[layer_number], layer_Dims[layer_number - 1]))
            else:
                layer['W'] = np.random.randn(layer_Dims[layer_number], layer_Dims[layer_number - 1])
            if layer_number == self.memory_vector_layer_location:
                layer['S'] = np.random.randn(layer_Dims[layer_number], layer_Dims[layer_number])
            layer['b'] = np.zeros((layer_Dims[layer_number], 1))
            self.Layers_weight.append(layer)
        # 如果提供了循环网络循环的次数，时间步，则在这里就可以先初始化数据网络结构
        if time_steps_number!=None:
            self.establish_network_time_steps_data_infrastructure(time_steps_number)

    # -----------------------初始化数据网络结构--------------------------
    def establish_network_time_steps_data_infrastructure(self, time_steps_number):
        self.time_steps_number = time_steps_number
        # 为每个时间步创建一个数据神经网络来存储数据
        for time_step in range(time_steps_number):
            t_Layers_data_list = []
            # 每个时间步网络的结构一样
            for layer in range(self.layers_number):
                layer = {}
                t_Layers_data_list.append(layer)
            self.T_Layers_data.append(t_Layers_data_list)

    # ------------------------前向传播----------------------------
    def forward_propagation(self, X_time_steps_batch, is_training_mode=True):
        '''
        :param self:
        :param X_time_steps_batch: (time_steps, batch_size, feature_dimension)
        :return:
        '''
        if is_training_mode:
            # 如果正在训练模型，则预先计算记忆层中 C_Z 中的部分元素 activation_gradient_function(Z)
            # 为使用梯度反向传播做准备
            memory_C_Z_activation_function_gradient_vector = np.empty((self.time_steps_number, 1), dtype=np.object)
        # 从网络第零层往最后一层输出层逐层计算。
        self.batch_size = X_time_steps_batch.shape[1]
        for layer_number in range(self.layers_number):
            if layer_number==0:
                # 用训练数据初始化 RNN 时间步网络数据结构的第 0 层
                self.store_time_steps_array_data(X_time_steps_batch, layer_index=0, data_type='A')
                continue
            elif layer_number!=self.memory_vector_layer_location:
                # 这里的维度变化是： A(time_steps, batch_size, feature_dimension_l -1 ) 矩阵乘以
                # W(feature_dimension_l,feature_dimension_l - 1).T  加上  b(feature_dimension_l，1)
                # Z(time_steps, batch_size, feature_dimension_l)
                Z_time_steps_batch = np.dot(self.get_time_steps_array_data(layer_index=layer_number-1, data_type='A'),
                                            self.Layers_weight[layer_number]['W'].T) + self.Layers_weight[layer_number]['b'].T
                self.store_time_steps_array_data(Z_time_steps_batch, layer_index=layer_number, data_type='Z')
                activation_function = self.activation_functions_dict[self.Layers_weight[layer_number]['activation_function_name']]
                A_time_steps_batch = activation_function(Z_time_steps_batch)
                self.store_time_steps_array_data(A_time_steps_batch, layer_index=layer_number, data_type='A')
            else: # 如果是在含有记忆向量的层
                for time_step in range(self.time_steps_number):
                    # 在时间步 0 时候，初始化记忆向量 self.T_Layers_data[time_step=-1][layer_number]['H'] = 0
                    if time_step==0:
                        H_zero_start = np.zeros((self.batch_size, self.Layers_dims[self.memory_vector_layer_location]))
                        self.T_Layers_data[time_step][layer_number]['Z'] = np.dot(
                            self.T_Layers_data[time_step][layer_number - 1]['A'],
                            self.Layers_weight[layer_number]['W'].T)  + \
                            np.dot(H_zero_start, self.Layers_weight[layer_number]['S'].T) + \
                               self.Layers_weight[layer_number]['b'].T
                    else:
                        self.T_Layers_data[time_step][layer_number]['Z'] = np.dot(self.T_Layers_data[time_step][layer_number-1]['A'],
                                                                         self.Layers_weight[layer_number]['W'].T) + \
                                                                  np.dot(self.T_Layers_data[time_step-1][layer_number]['H'],
                                                                         self.Layers_weight[layer_number]['S'].T) + \
                                                                  self.Layers_weight[layer_number]['b'].T
                    # 如果模型训练中，则把 activation_gradient_function(Z) 的值按照时间顺序保存在 memory_C_Z_activation_function_gradient_vector 中
                    if is_training_mode:
                        activation_gradient_function = self.activation_gradient_functions_dict[self.Layers_weight[layer_number]['activation_function_name']]
                        memory_C_Z_activation_function_gradient_vector[time_step][0] = activation_gradient_function(self.T_Layers_data[time_step][layer_number]['Z'])
                    activation_function = self.activation_functions_dict[self.Layers_weight[layer_number]['activation_function_name']]
                    self.T_Layers_data[time_step][layer_number]['A'] = activation_function(self.T_Layers_data[time_step][layer_number]['Z'])
                    self.T_Layers_data[time_step][layer_number]['H'] = self.T_Layers_data[time_step][layer_number]['A']
        A_time_steps_batch_L = self.get_time_steps_array_data(layer_index=-1, data_type='A')
        if is_training_mode:
            C_Z_activation_function_gradient_vector = \
                self.memory_C_Z_partial_gradient_vector_3_dims_to_2_matrix(memory_C_Z_activation_function_gradient_vector)
            return (A_time_steps_batch_L, C_Z_activation_function_gradient_vector)
        else:
            return A_time_steps_batch_L

    # 把 memory_C_Z_partial_gradient_vector_3_dims_to_2_matrix 转化为动态规划算法需要的维度格式
    def memory_C_Z_partial_gradient_vector_3_dims_to_2_matrix(self,memory_C_Z_activation_function_gradient_vector):
        # 注意按照上面步骤计算出的 memory_C_Z_activation_function_gradient_vector 维度是 (time_steps,1)
        # 但是它里面元素的维度是三维的 (1,batch_size, 记忆向量维度) 需要将它转换成 (batch_size, 记忆向量维度)
        # 才能交与动态规划矩阵进行计算
        C_Z_activation_function_gradient_vector = np.empty((self.time_steps_number, 1), dtype=np.object)
        for i in range(self.time_steps_number):
            C_Z_activation_function_gradient_vector[i][0] = memory_C_Z_activation_function_gradient_vector[i][0][0]
        return C_Z_activation_function_gradient_vector

    def delta_Psi_3_dims_to_Delta_Psi_matrix(self,Delta_Psi_3_dims):
        Delta_Psi_matrix = np.empty((self.time_steps_number, 1), dtype=np.object)
        for i in range(self.time_steps_number):
            Delta_Psi_matrix[i][0] = Delta_Psi_3_dims[i]
        return Delta_Psi_matrix

    def c_Z_gradient_matrix_to_Delta_Psi_3_dims(self,C_Z_memery_matrix):
        temp_list = []
        for i in range(self.time_steps_number):
            temp_list.append([C_Z_memery_matrix[i][0]])
        Delta_memery_layer = np.concatenate(temp_list, axis=0)
        return Delta_memery_layer

    def _count_C_W_S(self,t_step ,layer_index, data_type_need):
        Delta_time_steps_batch = self.get_time_steps_array_data(layer_index, 'Delta')
        if data_type_need=='A':
            A_S_time_steps_batch = self.get_time_steps_array_data(layer_index - 1, data_type_need)
        elif data_type_need=='H':
            A_S_time_steps_batch = self.get_time_steps_array_data(layer_index, data_type_need)
        return np.dot(Delta_time_steps_batch[t_step].T, A_S_time_steps_batch[t_step])

    def count_C_W_S_and_store(self, layer_index, data_type=''):
        if data_type=='C_W':
            data_type_need = 'A'
        elif data_type=='C_S':
            data_type_need = 'H'
        C_W_S_time_steps_list = list(map(self._count_C_W_S,[t for t in range(self.time_steps_number)],
                                         [layer_index] * self.time_steps_number,
                                         [data_type_need] * self.time_steps_number))
        C_W_S = reduce(lambda x,y:x+y,C_W_S_time_steps_list)
        self.Layers_weight[layer_index][data_type] = C_W_S


    # ------------------------反向传播----------------------------
    def back_propagation(self, Delta_L, C_Z_activation_function_gradient_vector):
        self.store_time_steps_array_data(Delta_L, layer_index=-1, data_type='Delta')
        for layer_number in range(self.layers_number-1, 0, -1):
            # 如果不是最后一层且不是循环层
            if layer_number!=(self.layers_number-1) and layer_number!=self.memory_vector_layer_location:
                activation_function = self.activation_functions_dict[self.Layers_weight[layer_number]['activation_function_name']]
                Delta_time_steps_batch = np.dot(self.get_time_steps_array_data(layer_index=layer_number+1, data_type='Delta'),
                                                self.Layers_weight[layer_number+1]['W']) * activation_function(
                                                self.get_time_steps_array_data(layer_index=layer_number, data_type='Z'))
                self.store_time_steps_array_data(Delta_time_steps_batch, layer_index=layer_number, data_type='Delta')
            elif layer_number==self.memory_vector_layer_location:
                S_matrix = self.Layers_weight[layer_number]['S']
                activation_function = self.activation_functions_dict[self.Layers_weight[layer_number]['activation_function_name']]
                Delta_Psi_3_dims = np.dot(self.get_time_steps_array_data(layer_index=layer_number+1, data_type='Delta'),
                                                self.Layers_weight[layer_number+1]['W']) * activation_function(
                                                self.get_time_steps_array_data(layer_index=layer_number, data_type='Z'))
                Delta_Psi_matrix = self.delta_Psi_3_dims_to_Delta_Psi_matrix(Delta_Psi_3_dims)
                DP_RNN_BP = Dynamic_programming_accelerate_RNN_back_propagation(Delta_Psi_matrix,
                                                                                C_Z_activation_function_gradient_vector,
                                                                                S_matrix)
                #print('开始启用动态规划对 RNN 反向传播进行加速！')
                C_Z_memery_matrix = DP_RNN_BP.calculating_cyclic_layer_gradient()
                #print('动态规划完成循环层梯度加速计算！')
                Delta_memery_layer = self.c_Z_gradient_matrix_to_Delta_Psi_3_dims(C_Z_memery_matrix)
                self.store_time_steps_array_data(Delta_memery_layer, layer_index=layer_number, data_type='Delta')
            else:
                pass
            Delta_time_steps_batch = self.get_time_steps_array_data(layer_number, 'Delta')
            C_b = (1.0 / self.batch_size) * np.sum(Delta_time_steps_batch,axis=(0,1)).reshape((-1, 1))
            self.Layers_weight[layer_number]['C_b'] = C_b
            self.count_C_W_S_and_store(layer_index=layer_number, data_type='C_W')
            if layer_number==self.memory_vector_layer_location:
                self.count_C_W_S_and_store(layer_index=layer_number, data_type='C_S')

    # ----------------------小批量梯度下降-------------------------
    def gradient_descent(self, learning_rate):
        # 输入层没有参数，所以从第一层开始。
        for layer_index in range(1, self.layers_number):
            self.Layers_weight[layer_index]['b'] -= learning_rate * self.Layers_weight[layer_index]['C_b']
            self.Layers_weight[layer_index]['W'] -= learning_rate * self.Layers_weight[layer_index]['C_W']
            if layer_index==self.memory_vector_layer_location:
                self.Layers_weight[layer_index]['S'] -= learning_rate * self.Layers_weight[layer_index]['C_S']

    # ----------------------训练和评估模型------------------------
    def training_and_evaluation_model(self, X_train, Y_train, epochs, loss_faction_name,
                                      learning_rate, evaluation_model_per_epochs):
        self.loss_faction_name = loss_faction_name
        for epoch in range(epochs):
            # 前向传播
            A_time_steps_batch_L, memory_C_Z_activation_function_gradient_vector = \
                self.forward_propagation(X_train, is_training_mode=True)
            # 损失函数对输出层的梯度
            Delta_L = A_time_steps_batch_L - Y_train
            # 反向传播
            self.back_propagation(Delta_L, memory_C_Z_activation_function_gradient_vector)
            # 梯度下降
            self.gradient_descent(learning_rate)
            if epoch % evaluation_model_per_epochs == 0:
                # 计算损失
                Loss_value = self.rnn_cross_entropy_loss_function(Y_train)
                print("Loss_value:\t\t{:.5f}".format(Loss_value))
                # 计算准确率
                accuracy_rate = self.rnn_accuracy_rate(Y_train)
                print("accuracy_rate:\t\t\t{:.5f}".format(accuracy_rate))


    # -------------------时间步数据网络数据存取函数-----------------------
    # 把数据存储到对应的时间步数据网络
    #def store_time_steps_array_data(self, K_time_steps_batch, layer_index=None, data_type=""):
    #    K_time_steps_batch_list = np.array_split(K_time_steps_batch, self.time_steps_number, axis=0)
    #    for time_step in range(self.time_steps_number):
    #        self.T_Layers_data[time_step][layer_index][data_type] = K_time_steps_batch_list[time_step]
    # 效果与上面注释函数一样，但是通过 map 函数可以对各个时间步并行存数据，速度更快。
    def _store_data_map(self, k_time_steps_batch, time_step, layer_index, data_type):
        self.T_Layers_data[time_step][layer_index][data_type] = k_time_steps_batch
        return True

    def store_time_steps_array_data(self, K_time_steps_batch, layer_index=None, data_type=""):
        K_time_steps_batch_list = np.array_split(K_time_steps_batch, self.time_steps_number, axis=0)
        # map() 函数前面加 list() 是为了让 map() 函数执行
        list(map(self._store_data_map, K_time_steps_batch_list,[t for t in range(self.time_steps_number)],
                 [layer_index] * self.time_steps_number,
                 [data_type] * self.time_steps_number))

    # 取所有时间步数据网络的某一层某种数据
    #def get_time_steps_array_data(self, layer_index=None, data_type=""):
    #    temp_data_list = []
    #    for time_step in range(self.time_steps_number):
    #        temp_data_list.append(self.T_Layers_data[time_step][layer_index][data_type])
    #    return np.concatenate(temp_data_list, axis=0)
    def get_time_steps_array_data(self, layer_index=None, data_type=""):
        # 效果与上面注释函数一样，但是通过 map 函数可以对各个时间步并行取数据，速度更快。
        temp_map_data = map(lambda time_step,layer_index,data_type:self.T_Layers_data[time_step][layer_index][data_type],
                            [t for t in range(self.time_steps_number)],
                            [layer_index] * self.time_steps_number,
                            [data_type] * self.time_steps_number)
        # 取出来的维度是 (time_steps, batch_size, feature_dimension)
        return np.concatenate(list(temp_map_data), axis=0)



    # ---------------------激活函数及其导数------------------------
    # 恒等激活函数
    def equal(slef, F):
        return F
    # 恒等激活函数的导数
    def equal_gradient(slef, F):
        return np.ones_like(F)
    # sigmoid 激活函数
    def sigmoid(self, F):
        return 1.0 / (1 + np.exp(-F))
    # sigmoid 激活函数的导数
    def sigmoid_gradient(self, F):
        return self.sigmoid(F) * (1 - self.sigmoid(F))
    # Relu 激活函数
    def relu(self, F):
        return np.maximum(F, 0, F)
    # Relu 激活函数的导数
    def relu_gradient(self, F):
        return np.where(F > 0, 1, 0)
    # Leaky Relu 激活函数
    def leaky_relu(self, F):
        # 有论文推荐用较小的 0.01 值，实践中似乎 0.2 更好
        leaky_value = 0.2
        return np.maximum(leaky_value * F, F, F)
    # Leaky Relu 激活函数的导数
    def leaky_relu_gradient(self, F):
        leaky_value = 0.2
        return np.where(F > 0, 1, leaky_value)
    # Softmax 激活函数
    def softmax(self, F):
        exp = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    # ---------------------损失函数------------------------
    # RNNs 交叉熵损失函数
    def rnn_cross_entropy_loss_function(self, Y_train):
        A_L = self.get_time_steps_array_data(-1, 'A')
        def _entropy_loss_map(t_Y_train, t_A_L):
            return (-1) * np.mean(np.sum(t_Y_train * np.log(t_A_L), axis=1, keepdims=True))
        T_Loss_list = map(_entropy_loss_map, Y_train, A_L)
        Loss_value = reduce(lambda x, y: x + y, T_Loss_list)
        return Loss_value

    # ---------------------分类决策函数------------------------
    # RNNs softmax 决策函数
    def rnn_softmax_decision(self,):
        A_L = self.get_time_steps_array_data(-1, 'A')
        def _softmax_decision_map(t_A_L):
            # 取矩阵每一行最大的值
            max_index = np.max(t_A_L, axis=1, keepdims=True)
            # 把最大的值的位置的值设置为 1，其余的设置为 0。
            t_Y_hat_time_steps = np.where(t_A_L == max_index, 1, 0)
            return [t_Y_hat_time_steps]
        Y_hat_time_steps_batch_list = list(map(_softmax_decision_map, A_L))
        Y_hat_time_steps_batch = np.concatenate(Y_hat_time_steps_batch_list, axis=0)
        return Y_hat_time_steps_batch

    # -------------------模型评价函数-----------------------
    # RNNs softmax 模型准确率函数
    def rnn_accuracy_rate(self,Y_train_time_steps_batch):
        Y_hat_time_steps_batch = self.rnn_softmax_decision()
        def _rnn_accuracy_rate_map(t_Y_train, t_Y_hat):
            accuracy_rate = 0.0
            # 比较 Y_hat 和 Y 矩阵的每一行
            res_D = np.mean(np.where(t_Y_hat == t_Y_train, 1, 0), axis=1, keepdims=True)
            accuracy_rate += np.mean(np.where(res_D == 1, 1, 0))
            return accuracy_rate
        accuracy_rate_list = map(_rnn_accuracy_rate_map,Y_train_time_steps_batch, Y_hat_time_steps_batch)
        accuracy_rate_value = reduce(lambda x, y: x + y, accuracy_rate_list) / self.time_steps_number
        return accuracy_rate_value


    # ---------------------杂项函数------------------------
    # 显示循环神经网络权重结构和数据结构
    def show_network_structure(self, all_time_steps_data_structure=False):
        for layer_number in range(1, self.layers_number):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('layer_number_index:\t\t',layer_number)
            print('activation_function:',self.Layers_weight[layer_number]['activation_function_name'])
            print('W shape:\t\t\t',self.Layers_weight[layer_number]['W'].shape)
            if layer_number==self.memory_vector_layer_location:
                print('S shape:\t\t\t', self.Layers_weight[layer_number]['S'].shape)
            print('b shape:\t\t\t',self.Layers_weight[layer_number]['b'].shape)
        print('\n')
        if self.time_steps_number is None:
            for t in range(self.time_steps_number):
                if not all_time_steps_data_structure:
                    print('*********************************')
                    print("There are {} time steps,\ntheir data structure is the same.".format(self.time_steps_number))
                    print('*********************************')
                else:
                    print('*********************************')
                    print('time step number:\t\t\t', t)
                    print('*********************************')
                for layer_number in range(self.layers_number):
                    print('layer_number_index: \t\t', layer_number)
                    for k in self.T_Layers_data[t][layer_number].keys():
                        print("{} shape:\t\t\t{}".format(k,self.T_Layers_data[t][layer_number][k].shape))
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                if not all_time_steps_data_structure:
                    break

