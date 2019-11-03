#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #3D plot


# 经验势的表示函数,距离过近取为无穷大,过远取为0

# In[2]:


def emperical_potential(r1,r2,a=5,b=10,d_min=0.8,d_max=2.2):
    """Calculate the potential between 2 given atoms. The potential is derived from Van der Waals force.
    
    Args:
        r1,r2: Coordinates of atoms,1-D list-like. 2D and 3D situations are both accepted.
        a,b: Parameters of the potential expression with default value 5 and 10 respectively.
        
    Returns:
        If distance > d_max, the potential value equals to zero.
        If distance < d_min, the potential value approaches infinity.
    """
    d = distance.euclidean(r1,r2)
    if d > d_max:
        return 0
    elif d < d_min:
        return float("inf")
    else:
        return a / (d ** 12) - b / (d ** 6)


# 原子类,包含原子坐标

# In[3]:


class Atom():
    def __init__(self,coords):
        self.coords = coords


# 原子点集类,含维数、总势能、原子个数、坐标范围等参数,并有随机生成、输入列表生成、势能计算等方法

# In[4]:


class AtomCollection():
    def __init__(self):
        """
        Args:
        coords_range: suppose (x, ...) is the coordinate of an atom while range is (a,b), then a<x<b, ...
        """
        self.atoms = []
        self.num = 0
        self.dimension = 0
        self.potential = 0
        self.coords_range=(-5,5)
        
    def random_generate(self,size,dimension=2):
        """generate a collection of atoms by using random numbers
        
        Args:
            size: the number of atoms in self.atoms
            dimension: 2 or 3
        """
        self.dimension = dimension
        temp_coords = np.random.uniform(self.coords_range[0],self.coords_range[1],
                                        dimension).tolist()  #1*dimension shape-like list
        self.num +=1
        self.atoms.append(Atom(temp_coords))
        while self.num < size:
            temp_coords = np.random.uniform(self.coords_range[0],self.coords_range[1],
                                            dimension).tolist()
            temp_potential = [emperical_potential(temp_coords,atom.coords) for atom in self.atoms]
            if float("inf") in temp_potential:
                continue  #if too close, then regenerate
            else:
                self.num +=1
                self.atoms.append(Atom(temp_coords))
                    
    def list_generate(self,coords_list):
        """generate a collection of atoms by using a given list
        
        Args:
            coords_list: shape like [[1,1,...],[2,2,...],...]
        """
        if coords_list == []:
            return
        for coords in coords_list:
            self.atoms.append(Atom(coords))
        self.num = len(coords_list)
        self.dimension = len(coords_list[0])
    
    def potential_cal(self):
        """calculate the total potential of the atom collection"""        
        for atom in self.atoms: #对每个原子进行循环
            for atom1 in self.atoms: #计算每个原子的总势能，再相加
                if atom1 is not atom:
                    self.potential += emperical_potential(atom.coords,atom1.coords)
        self.potential /= 2.0
        
    def plot_atoms(self):
        """plot all the atoms on one figure"""
        fig = plt.figure()
        
        if self.dimension == 2:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=self.coords_range, ylim=self.coords_range)
            for atom in self.atoms:
                ax.scatter(atom.coords[0],atom.coords[1], color = 'black')
            
        if self.dimension == 3:
            ax = Axes3D(fig)
            for atom in self.atoms:
                ax.scatter(atom.coords[0], atom.coords[1],atom.coords[2])
            ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
            ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
            ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})
    
    def clear_collection(self):
        """erase all the coordinates"""
        self.atoms = []
        self.num = 0
        self.dimension = 0
        self.potential = 0


# In[5]:


if __name__ == "__main__":
    test_set = AtomCollection()
    test_set.random_generate(10,2)
    test_set.potential_cal()
    test_set.plot_atoms()


# In[6]:


if __name__ == "__main__":
    test_set = AtomCollection()
    test_set.random_generate(10,3)
    test_set.potential_cal()
    test_set.plot_atoms()


# In[ ]:





# In[ ]:




