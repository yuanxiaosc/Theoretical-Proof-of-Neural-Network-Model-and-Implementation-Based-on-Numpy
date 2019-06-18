from RNN.hans_on_recurrent_neural_network import Recurrent_neural_network
import random
import numpy as np

# ---------------------------导入数据-------------------------------
#-------------------------------------------------------------------

time_step = 3
sample_feature_dimension = 3
batch_size = 10

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




# ----------------------------构建模型------------------------------
#-------------------------------------------------------------------
Net = Recurrent_neural_network()

# 样本的特征维度
sample_feature_dimension = 3
# 神经网络每一层的维度
layer_Dims = [4, 5, 4, 3]
# 神经网络每一层使用的激活函数
activation_function_names = ['sigmoid', 'sigmoid', 'sigmoid','softmax']
# 神经网络使用的损失函数
network_loss_faction_name = 'multi_classification_cross_entropy_loss_function'
# 神经网络记忆向量所在层数，默认是在输出层之前
memory_vector_layer_location = 3

Net.creating_network_weight_infrastructure(
    sample_feature_dimension, layer_Dims, activation_function_names,
    time_steps_number=3)



#----------------------------训练和评估模型--------------------------
#-------------------------------------------------------------------
epochs = 10
learning_rate = 0.01
evaluation_model_per_epochs = 1

X_train, Y_train = X_data, Y_data
time_steps_number = X_train.shape[0]

#Net.establish_network_time_steps_data_infrastructure(time_steps_number)

Net.training_and_evaluation_model(X_train,Y_train,epochs,'',
                                  learning_rate, evaluation_model_per_epochs)


#Net.show_network_structure()
