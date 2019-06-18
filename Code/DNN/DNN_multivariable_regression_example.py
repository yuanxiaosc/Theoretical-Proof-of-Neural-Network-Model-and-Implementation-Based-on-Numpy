from DNN.hans_on_feedforward_neural_network import Feedforward_neural_network
import numpy as np

Net = Feedforward_neural_network()

#--------------------------多元回归实验-----------------------------

# ---------------------------准备数据-------------------------------
#-------------------------------------------------------------------
# 20 维到 3 维的转换
X_data = np.random.uniform(0, 100, size=(1000, 20))
W = np.random.random(size=(20, 3))
Y_data =  np.dot(X_data, W)

# 给标签加上高斯白噪声，使之成为非线性关系
Y_data = Y_data + np.random.normal(0, 10, size=Y_data.shape)

sample_numbers = X_data.shape[0]

shuffle_index = np.random.permutation(sample_numbers)
X_train, Y_train = X_data[shuffle_index], Y_data[shuffle_index]

print('X_train:\t', X_train.shape)
print('Y_train:\t', Y_train.shape)

# ----------------------------构建模型------------------------------
#-------------------------------------------------------------------
# 样本的特征维度
sample_feature_dimension = 20
# 神经网络每一层的维度
layer_Dims = [30, 15, 3]
# 神经网络每一层使用的激活函数
activation_function_names = ['sigmoid', 'relu', 'None']
# 神经网络使用的损失函数
network_loss_faction_name = 'multivariable_regression_loss_function'

Net.creating_network_infrastructure(sample_feature_dimension, layer_Dims, activation_function_names)

#----------------------------训练和评估模型--------------------------
#-------------------------------------------------------------------
epochs = 10
learning_rate = 0.00001
batch_size = 100
evaluation_model_per_epochs = 1

# 重新初始化模型权重参数
#Net.initialization_model_again()

Net.training_and_evaluation_model(X_train,Y_train,network_loss_faction_name,\
                                  epochs,learning_rate, batch_size,\
                                 evaluation_model_per_epochs)

#----------------------------查看模型参数--------------------------
#------------------------------------------------------------------
model_info_dict = Net.export_model_parameters()
# 样本的特征维度
print(model_info_dict['sample_feature_dimension'])
# 神经网络每一层的维度
print(model_info_dict['neurons_numbers'])
# 神经网络每一层使用的激活函数
print(model_info_dict['activation_function_names'])
# 神经网络使用的损失函数
print(model_info_dict['loss_faction_name'])
# 神经网络每一层的参数值，例如第一层
#print(model_info_dict['parameter_values'][1])