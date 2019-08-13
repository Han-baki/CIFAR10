#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import numpy as np

def load_cifar10(path = 'cifardata/cifar-10-batches-py/'):
  
    import os
    import pickle
    import numpy as np

    def unpickle(file):
        import pickle
        with open(file,'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict

    def load_data(path, file):
        
        return unpickle(path + file)
      
    cifar_data_1 = load_data(path,'data_batch_1')
    cifar_data_2 = load_data(path,'data_batch_2')
    cifar_data_3 = load_data(path,'data_batch_3')
    cifar_data_4 = load_data(path,'data_batch_4')
    cifar_data_5 = load_data(path,'data_batch_5')
    cifar_test = load_data(path,'test_batch')
    cifar_data = [cifar_data_1,cifar_data_2,cifar_data_3,cifar_data_4,cifar_data_5,cifar_test]
    datas = []
    labels = []

    for i in range(6):
        datas.append(cifar_data[i][b'data'])
        labels.append(cifar_data[i][b'labels'])

    input_datas = []
    input_labels = []

    for i in range(6):
        input_datas.append(np.transpose(np.reshape(datas[i],[-1,3,32,32]), [0,2,3,1])/255)
        input_labels.append(np.array(labels[i]))
    
    print('image shape: ', np.shape(input_datas))
    print('label shape: ', np.shape(input_labels))
    return input_datas, input_labels
  

