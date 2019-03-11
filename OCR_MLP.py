
# coding: utf-8

# In[21]:


import sys
import os

import cv2
import numpy as np


# In[39]:


input_file='Desktop/summer project/letter.data'
img_height=16
img_width=8
img_resize_factor=22


# In[ ]:


labels=[]
with open(input_file,'r') as f:
    for line in f.readlines():
        data=np.array([255*float(x) for x in line.split('\t')[6:-1]])
        image_label=line.split('\t')[1]
        if image_label not in labels:
            labels.append(image_label)
        image=np.reshape(data,(img_height,img_width))
        image_scaled=cv2.resize(image,None,fx=img_resize_factor,fy=img_resize_factor)
        cv2.imshow('IMG',image_scaled)
        print('Label : ',image_label)
        print(len(data))
        wkey=cv2.waitKey()
        if wkey==27:
           break


# In[ ]:


num_data = 50
orig_labels = 'omandig'
num_orig_labels = len(orig_labels)

num_train = int(0.9*num_data)
num_test = num_data - num_train

start = 6
end = -1


# In[ ]:


data = []
labels = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in orig_labels:
            continue
        label = np.zeros((num_orig_labels,1))
        label[orig_labels.index(list_vals[1])]=1
        labels.append(label)
        
        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)
        
        if len(data) >= num_data:
            break
            


# In[ ]:


data_r=(np.array(data).reshape(50,128))
labels_r=np.array(labels).reshape(50,num_orig_labels)
labels_r[0].shape

data_train=data_r[:num_train]
data_test=data_r[num_train:]
labels_train=labels_r[:num_train]
labels_test=labels_r[num_train:]


# In[130]:


from sklearn.neural_network import MLPClassifier as MLP
nn=MLP(hidden_layer_sizes=(128,16,num_orig_labels),max_iter=20000,tol=0.01)
nn=nn.fit(data_train,labels_train)
nn.score(data_test,labels_test)

