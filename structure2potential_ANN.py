#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #3D plot


# In[2]:


from structure2potential import *


# 神经网络模拟

# In[3]:


import tensorflow as tf


# In[4]:


def get_batch(atom_collection_set,batch_size):
    temp = np.array(atom_collection_set)
    index = np.random.randint(0, len(atom_collection_set), batch_size)
    temp = temp[index].tolist()
    coordinates = []
    potentials = []
    for samples in temp:
        coordinates.append([sample.coords for sample in samples.atoms])
        potentials.append(samples.potential)
    return np.array(coordinates), np.array(potentials)


# In[5]:


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100)
        self.dense2 = tf.keras.layers.Dense(units=100)
        self.dense3 = tf.keras.layers.Dense(units=10)
        self.dense4 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):         
        x = self.flatten(inputs)    # [batch_size, atom_size*dimension]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        x = self.dense3(x)
        x = self.dense4(x)
        output = x
        return output


# In[6]:


#超参数hyperparameters
num_epoch = 0.5
batch_size = 5
rate = 0.001
atom_size = 20
num_train_data = 1000
num_test_data = 50


# In[7]:


#建模 训练
atoms_collections = []
for i in range(num_test_data):
    #生成训练数据集
    temp = AtomCollection()
    temp.random_generate(atom_size)
    temp.potential_cal()
    atoms_collections.append(temp)
    
model = MLP()
optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
num_batches = int(num_train_data//batch_size*num_epoch)

for batch_index in range(num_batches):
    X, y = get_batch(atoms_collections,batch_size)  #X: (batch_size,atom_size,dimension), y: (batch_size,)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = 0.5*tf.reduce_sum(tf.square(y_pred-y))
    #print("batch {}: loss {}".format(batch_index,loss.numpy()))
    grad = tape.gradient(loss,model.variables)#model.variables直接调用模型变量
    optimizer.apply_gradients(grads_and_vars=zip(grad,model.variables))


# In[8]:


#评估
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
atoms_test = []
energy_delta = []
num_batches = int(num_test_data//batch_size)
for i in range(num_train_data):
    #生成测试数据集
    temp = AtomCollection()
    temp.random_generate(atom_size)
    temp.potential_cal()
    atoms_test.append(temp)

for batch_index in range(num_batches):
    start,end = batch_index*batch_size,(batch_index+1)*batch_size
    X, y_test = get_batch(atoms_test,batch_size)
    y_pred = model.predict(X)
    for i in range(batch_size):
        energy_delta.append(abs(y_test[i] - y_pred[i][0]/y_test[i])*100)


# In[9]:


energy_delta.sort()


# In[10]:


print(energy_delta)


# In[ ]:




