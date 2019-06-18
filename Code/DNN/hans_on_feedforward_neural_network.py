'''
hans_on_feedforward_neural_network.py

前馈神经网络采用小批量随机梯度下降算法。

为什么要写此模块：
1. 为了深刻理解神经网络进而理解深度学习；
2. 神经网络框架源码读不懂，还是自己实现吧；

功能：实现了 Logistic 二分类，Softmax 多分类和多元回归算法，具体见样例代码。

说明：
1. 为了提高运算效率，前馈神经网络基于 numpy 库来做矩阵运算，很少使用 for 循环；
2. 未来会逐步添加一些新的神经网络功能，比如：不同的初始化方法、正则化等...欢迎读者来更新此代码；
3. 目前该网络存在数值不稳定性，建议不要把 “relu” 家族系列激活函数放在神经网络第一层，容易造成数值不稳定，
    放在其它层效果很好。而且目前 Softmax 和交叉熵是分开实现的，容易导致数值不稳定，未来会写成一个函数；
4. 毕竟我只是为了尽可能的展示原理，很少考虑代码如何更高效，所以欢迎读者来优化这份代码，尽请修改吧！
'''

# Third-party libraries
import numpy as np

# 基于Numpy的前馈神经网络
class Feedforward_neural_network(object):

    def __init__(self):
        # ---------------------前馈神经网络基本参数------------------------
        # 样本的特征维度
        self.sample_feature_dimension = None
        # 神经网络每一层的维度
        self.layer_Dims = None
        # 神经网络每一层使用的激活函数
        self.activation_function_names = None
        # 神经网络使用的损失函数
        self.network_loss_faction_name = None
        # 存储网络参数的列表
        self.Layers = []
        # ---------------------函数名与函数对象映射字典------------------------
        # 激活函数映射字典
        self.activation_functions_dict = {'sigmoid': self.sigmoid, 'relu':self.relu,
                                          'leaky_relu':self.leaky_relu, 'tanh':self.tanh,
                                          'softmax': self.softmax, 'None': self.equal}

        # 激活函数的导数函数的映射字典
        self.activation_gradient_functions_dict = {'sigmoid': self.sigmoid_gradient, 'tanh':self.tanh_gradient,'relu':self.relu_gradient,
                                                   'leaky_relu':self.leaky_relu_gradient,'softmax': None,
                                                   'None': self.equal_gradient}
        # 损失函数映射字典
        self.Loss_faction_dict = {
            'multi_classification_cross_entropy_loss_function': self.multi_classification_cross_entropy_loss_function,
            'binary_classification_logistic_loss_function': self.binary_classification_logistic_loss_function,
            'multivariable_regression_loss_function': self.multivariable_regression_loss_function}

    # ----------------------初始化网络结构------------------------
    # 创建网络结构的函数
    def creating_network_infrastructure(self, sample_feature_dimension, layer_Dims, activation_function_names):
        # 把输入层作为第 0 层，同时把它的维度插入维度数组中
        layer_Dims.insert(0, sample_feature_dimension)
        for layer_index in range(len(layer_Dims)):
            layer = {}
            #第0层是输入层，不需要激活函数和权重矩阵
            if layer_index == 0:
                self.Layers.append(layer)
                continue
            layer['activation_function_name'] = activation_function_names[layer_index - 1]
            # 因为 relu 函数会把不大于 0 的值 设置为 0，导致模型在开始时候较难训练
            # 所以当某一层使用 relu 作为激活函数时，该层权重矩阵使用大于 0 值的初始化方法
            if layer['activation_function_name']=='relu' or layer['activation_function_name']=='leaky_relu':
                layer['W'] = np.random.uniform(np.exp(-10), 1, size=(layer_Dims[layer_index], layer_Dims[layer_index - 1]))
            else:
                layer['W'] = np.random.randn(layer_Dims[layer_index], layer_Dims[layer_index - 1])
            layer['b'] = np.zeros((layer_Dims[layer_index], 1))
            self.Layers.append(layer)

    # ------------------------前向传播----------------------------
    def forward_propagation(self, X_batch):
        # 从网络第一层往最后一层输出层逐层计算。
        for layer_index, layer in enumerate(self.Layers):
            # 把样本输入直接作为激活之后的值。
            if layer_index == 0:
                layer['A'] = X_batch
            else:
                layer['Z'] = np.dot(self.Layers[layer_index - 1]['A'], np.transpose(layer['W'])) + np.transpose(layer['b'])
                activation_function = self.activation_functions_dict[layer['activation_function_name']]
                layer['A'] = activation_function(layer['Z'])
        return self.Layers[-1]['A']

    # ------------------------反向传播----------------------------
    def back_propagation(self, Delta_L):
        # 获取最后一层的误差Delta。
        self.Layers[-1]['Delta'] = Delta_L
        # 误差从最后一层往第一层遍历网络，不包括第 0 层（输入层）。
        for layer_index in range(len(self.Layers) - 1, 0, -1):
            layer = self.Layers[layer_index]
            # 如果不是最后一层
            if layer_index != len(self.Layers) - 1:
                activation_gradient_function = self.activation_gradient_functions_dict[layer['activation_function_name']]
                layer['Delta'] = np.dot(self.Layers[layer_index + 1]['Delta'],
                                        self.Layers[layer_index + 1]['W']) * activation_gradient_function(layer['Z'])
            _batch_size = Delta_L.shape[0]
            layer['C_b'] = (1.0 / _batch_size) * np.transpose((np.sum(layer['Delta'], axis=0, keepdims=True)))
            layer['C_W'] = (1.0 / _batch_size) * np.dot(np.transpose(layer['Delta']), self.Layers[layer_index - 1]['A'])

    # ----------------------小批量梯度下降-------------------------
    def gradient_descent(self, learning_rate):
        # 输入层没有参数，所以从第一层开始。
        for layer in self.Layers[1:]:
            layer['W'] = layer['W'] - learning_rate * layer['C_W']
            layer['b'] = layer['b'] - learning_rate * layer['C_b']

    # ----------------------训练和评估模型------------------------
    def training_and_evaluation_model(self, X_train, Y_train, network_loss_faction_name,
                                      epochs, learning_rate, batch_size, evaluation_model_per_epochs):
        self.network_loss_faction_name = network_loss_faction_name
        for epoch in range(epochs):
            batch_index = 0
            for batch in range(X_train.shape[0] // batch_size):
                X_batch = X_train[batch_index: batch_index + batch_size]
                Y_batch = Y_train[batch_index: batch_index + batch_size]
                batch_index = batch_index + batch_size
                # 前向传播
                self.forward_propagation(X_batch)
                Last_layer_output =  self.Layers[-1]['A']
                loss_f =  self.Loss_faction_dict[network_loss_faction_name]
                # 计算损失
                Loss = loss_f(Y_batch, Last_layer_output)
                # 损失函数对输出层的梯度
                if network_loss_faction_name == 'multi_classification_cross_entropy_loss_function' \
                        or network_loss_faction_name == 'binary_classification_logistic_loss_function':
                    Delta_L = Last_layer_output - Y_batch
                elif network_loss_faction_name == 'multivariable_regression_loss_function':
                    Delta_L = 2 * (Last_layer_output - Y_batch)
                else:
                    Delta_L = None
                # 反向传播
                self.back_propagation(Delta_L)
                # 梯度下降
                self.gradient_descent(learning_rate)
            if epoch % evaluation_model_per_epochs == 0:
                if network_loss_faction_name == 'multi_classification_cross_entropy_loss_function':
                    accuracy_rate = self.softmax_classified_accuracy_rate(X_train, Y_train)
                elif network_loss_faction_name == 'binary_classification_logistic_loss_function':
                    accuracy_rate = self.logistic_binary_classified_accuracy_rate(X_train, Y_train)
                elif network_loss_faction_name == 'multivariable_regression_loss_function':
                    accuracy_rate = None
                    matrix_cosine_similarity = self.multivariable_matrix_cosine_similarity(X_train, Y_train)
                if accuracy_rate != None:
                    print('accuracy_rate:\t\t {:.5f}'.format(accuracy_rate))
                if network_loss_faction_name == 'multivariable_regression_loss_function':
                    print('matrix_cosine_similarity:\t\t {:.5f}'.format(matrix_cosine_similarity))
                print('Loss:\t\t{:.5f}'.format(Loss))

    # ---------------------激活函数及其导数------------------------
    # 恒等激活函数
    def equal(slef, F):
        return F
    # 恒等激活函数的导数
    def equal_gradient(slef, F):
        return np.ones_like(F)
    # sigmoid 激活函数
    def sigmoid(self,F):
        return 1.0 / (1 + np.exp(-F))
    # sigmoid 激活函数的导数
    def sigmoid_gradient(self, F):
        return self.sigmoid(F) * (1 - self.sigmoid(F))
    # tanh 激活函数
    def tanh(self, F):
        return np.tanh(F)
    # tanh 激活函数的导数
    def tanh_gradient(self, F):
        return 1.0 - np.tanh(F)**2
    # Relu 激活函数
    def relu(self, F):
        return np.maximum(F, 0, F)
    # Relu 激活函数的导数
    def relu_gradient(self, F):
        return np.where(F>0, 1, 0)
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
    #def unstable_softmax(slef, F):
    #    return np.exp(F) / np.sum(np.exp(F), axis=1, keepdims=True)

    #---------------------损失函数------------------------
    # 多元回归损失函数
    def multivariable_regression_loss_function(self, Y_batch, A_L):
        return np.mean(np.sum((A_L - Y_batch) ** 2, axis=1, keepdims=True))

    # 多分类交叉熵损失函数
    def multi_classification_cross_entropy_loss_function(self, Y_batch, A_L):
        # log(x) 中 x 不能为 0，暂时办法：把 0 换成一个很小的数字 np.exp(-30)
        #A_L = np.where(A_L==0, np.exp(-30), A_L)
        return (-1) * np.mean(np.sum(Y_batch * np.log(A_L), axis=1, keepdims=True))

    # 二分类逻辑斯蒂损失函数
    def binary_classification_logistic_loss_function(self, Y_batch, A_L):
        # log(x) 中 x 不能为 0，暂时办法：把 0 换成一个很小的数字 np.exp(-30)
        #A_L = np.where(A_L == 0, np.exp(-30), A_L)
        return (-1.0) * np.mean(
            np.dot(np.transpose(Y_batch), np.log(A_L)) + np.dot(1 - np.transpose(Y_batch), np.log(1 - A_L)))

    # ---------------------分类决策函数------------------------
    # softmax 分类决策函数
    def softmax_classified_decision(self, F):
        # 取矩阵每一行最大的值
        max_index = np.max(F, axis=1, keepdims=True)
        # 把最大的值的位置的值设置为 1，其余的设置为 0。
        one_hot_hat_Y = np.where(F == max_index, 1, 0)
        return one_hot_hat_Y

    # logistic 二分类决策函数
    def logistic_binary_classified_decision(self, F):
        # 大于 0.5 的取值为 1 否则为 0。
        Y_hat = np.where(F >= 0.5, 1, 0)
        return Y_hat

    # -------------------模型评价函数-----------------------
    # 计算矩阵余弦相似度
    def multivariable_matrix_cosine_similarity(self, X, Y):
        # 前向传播
        Last_layer_output = self.forward_propagation(X)
        # 预测函数
        Y_hat = Last_layer_output
        # 比较 Y_hat 和 Y 矩阵的每一行, 计算每一行之间的余弦相似度
        res_D = np.sum(Y_hat * Y, axis=1, keepdims=True) / (
                np.sqrt(np.sum(Y_hat ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(Y ** 2, axis=1, keepdims=True)))
        return np.mean(res_D)

    # 统计 softmax 模型准确率函数
    def softmax_classified_accuracy_rate(self, X, Y):
        # 前向传播
        Last_layer_output = self.forward_propagation(X)
        # 预测函数
        Y_hat = self.softmax_classified_decision(Last_layer_output)
        # 比较 Y_hat 和 Y 矩阵的每一行,也可以用np.argmax()来比较最大值的位置是否一样
        res_D = np.mean(np.where(Y_hat == Y, 1, 0), axis=1, keepdims=True)
        return np.mean(np.where(res_D == 1, 1, 0))

    # 统计 Logistic 模型准确率函数
    def logistic_binary_classified_accuracy_rate(self, X, Y):
        # 前向传播
        Last_layer_output = self.forward_propagation(X)
        # 预测函数
        Y_hat = self.logistic_binary_classified_decision(Last_layer_output)
        # 比较 Y_hat 和 Y 矩阵的每一行
        return np.mean(np.where(Y_hat == Y, 1, 0))

    # -------------------相关辅助函数-----------------------
    # 初始化模型
    def initialization_model_again(self, ):
        for layer in self.Layers[1:]:
            layer['W'] = np.random.randn(layer['W'].shape[0], layer['W'].shape[1])
            layer['b'] = np.zeros((layer['b'].shape[0], 1))

    # 把序号标签矩阵转换成 one_hot 形式
    def y_to_onehot(self, Y):
        label_numbers = Y.shape[0]
        Y = Y.reshape(-1, 1)
        Y_unique = np.unique(Y)
        K = len(Y_unique)
        Y_one_hot = np.eye(K)[Y]
        return Y_one_hot.reshape(label_numbers, K)

    # 显示模型的结构
    def display_model_structure(self, ):
        for layer in self.Layers[1:]:
            W = layer['W']
            b = layer['b']
            ac = layer['activation_function_name']
            print('W:\n', W.shape)
            print('b:\n', b.shape)
            print('activation_function_name', ac)

    # 根据 Layers 反推模型参数
    def analysis_model_by_Layers(self,):
        neurons_numbers = []
        activation_function_names = []
        for layer in self.Layers[1:]:
            neurons_numbers.append(layer['b'].shape[0])
            activation_function_names.append(layer['activation_function_name'])
        return neurons_numbers,activation_function_names

    # 导出模型
    def export_model_parameters(self):
        neurons_numbers, activation_function_names = self.analysis_model_by_Layers()
        model_info_dict = {'parameter_values':self.Layers,
                           'sample_feature_dimension':self.Layers[0]['A'].shape[1],
                           'neurons_numbers':neurons_numbers,
                           'activation_function_names':activation_function_names,
                           'loss_faction_name':self.network_loss_faction_name}
        return model_info_dict
