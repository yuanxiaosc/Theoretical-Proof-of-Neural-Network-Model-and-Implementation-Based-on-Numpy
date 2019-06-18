from DNN.hans_on_feedforward_neural_network import Feedforward_neural_network
import numpy as np

Net = Feedforward_neural_network()


#-----------------------Logistic 二分类实验-------------------------

# ---------------------------导入数据-------------------------------
#-------------------------------------------------------------------
from sklearn import datasets

iris = datasets.load_iris()
X_data = iris["data"][:, (2, 3)]  # petal length, petal width
Y_data = (iris['target']==2).reshape(-1, 1) # Iris-Virginica
sample_numbers = X_data.shape[0]

# Y_train 已经是 onehot 矩阵，否则应该先转换
#Y_one_hot_data = y_to_onehot(Y_data)

shuffle_index = np.random.permutation(sample_numbers)
X_train, Y_train = X_data[shuffle_index], Y_data[shuffle_index]

print('X_train:\t', X_train.shape)
print('Y_train:\t', Y_train.shape)

# ----------------------------构建模型------------------------------
#-------------------------------------------------------------------
# 样本的特征维度
sample_feature_dimension = X_train.shape[1]
# 神经网络每一层的维度
layer_Dims = [3, 1]
# 神经网络每一层使用的激活函数
activation_function_names = ['sigmoid', 'sigmoid']
# 神经网络使用的损失函数
network_loss_faction_name = 'binary_classification_logistic_loss_function'

Net.creating_network_infrastructure(sample_feature_dimension, layer_Dims, activation_function_names)

#----------------------------训练和评估模型--------------------------
#-------------------------------------------------------------------
epochs = 100
learning_rate = 0.1
batch_size = 10
evaluation_model_per_epochs = 10

# 重新初始化模型权重参数
Net.initialization_model_again()

Net.training_and_evaluation_model(X_train, Y_train, network_loss_faction_name, epochs,\
                              learning_rate, batch_size, evaluation_model_per_epochs)

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
