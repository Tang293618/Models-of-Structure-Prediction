#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from structure2potential import *


# 参考文献:On-the-fly machine learning of atomic potential in density functional theory
# structure optimization

# In[3]:


def distance_cal(atom_set):
    dimension = atom_set.dimension
    radius_square = []
    for i in range(atom_set.num):
        for j in range(atom_set.num):
            temp = 0
            if i == j:
                pass
            else:
                for k in range(dimension):
                    temp += (atom_set.atoms[i].coords[k] - atom_set.atoms[j].coords[k])**2
            radius_square.append(temp)
    return radius_square


# In[4]:


def gfactor_cal(center_num,radius_square,r_l,r_delta):
    """Calculate the g_factor for i_th atom with radius ranging from (r_l,r_l+r_delta], i.e. g_i(r_l)
    
    Args:
        center_num: the serial number of the central atom(type: Atom),[0,total_num)
        radius_square: contains all the distance information in the corresponding atom collection
        r_l: the lower bound of radius
        r_delta: the increment of radius
        
    Returns:
        the value of g_i(r_l)
    
    """
    max_num = int(np.sqrt(len(radius_square)))
    volume = 4.0 / 3 * np.pi * ((r_l + r_delta)**3 - r_l**3)
    atom_set = []
    for i in range(max_num):
        if r_l**2 < radius_square[i + max_num * center_num] <= (r_l+r_delta)**2:
            atom_set.append(radius_square[i + max_num * center_num])
    #print(len(atom_set))
    gamma = 0.5 * len(atom_set) * (len(atom_set) - 1)
    g_value = 0
    if gamma == 0:
        pass
    else:
        for r_square in atom_set:
            g_value += 1.0 / r_square
        g_value *= (volume / (gamma * 4 * np.pi)) 
    return g_value


# In[5]:


def gvector_cal(r_min,r_max,r_delta,center_num,radius_square):
    """Calculate the g_vector for i_th atom, i.e. gi_vector = {gi(rmin), gi(rmin+rdelta), ..., gi(rmax)}
    
    """
    r_temp = r_min
    times=1
    g_vector = []
    while r_temp < r_max + r_delta:  #不用r_temp<=r_max作为条件是考虑到了计算机存储浮点数的精度问题
        g_temp = gfactor_cal(center_num,radius_square,r_temp,r_delta)
        g_vector.append(g_temp)
        r_temp += r_delta
    return np.array(g_vector)


# In[6]:


def my_kernel(X,Y,gamma=1):
    """The kernel function about to be used is similar with the gaussian function, except
    for that the distance is derived from a cosine distance rather than the euclidean one
    
    Args:
    X,Y: the type of those vectors are np.array(1,-1) 
    gamma: represents the coefficient 1/(2*sigma^2)
    """
    distance = 0.5 * (1 - np.dot(X,Y.T) / np.sqrt(np.dot(X,X.T)*np.dot(Y,Y.T)))
    return np.exp(- gamma * distance)


# In[7]:


r_delta = 0.2
r = 0.8
r_L = 6.0
atom_size = 50
vector_size = int((r_L - r) / r_delta) + 1
train_size = 20000


# In[ ]:


X = []
y = []
for j in range(train_size):
    atom_group = AtomCollection()
    atom_group.random_generate(size=atom_size)
    r_square = distance_cal(atom_group)
    f_vector = np.zeros(vector_size)
    for i in range(atom_group.num):
        f_vector += 0.5 * gvector_cal(r,r_L,r_delta,i,r_square)
    X.append(f_vector)
    atom_group.potential_cal()
    y.append(atom_group.potential)
        #print(gvector_cal(r,r_L,r_delta,i,r_square))
    #print(f_vector)  #1*vector_size shape-like array
X = np.array(X).reshape(-1,vector_size)
y = np.array(y).ravel()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


clf = KernelRidge(alpha=1.0,kernel=my_kernel)
clf.fit(X_train,y_train)


# In[ ]:


train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))


# In[ ]:


from plot_learning_curve import *
title = "Learning Curves (Kernel Ridge Regression)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(clf,title,X,y,cv=cv,n_jobs=4)
plt.show()


# In[ ]:




