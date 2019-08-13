#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


def noise(img):
    img_size = img.shape[:2]
    scale = np.random.randint(16)
#     noise = np.array(np.random.exponential(scale, img_size), dtype=np.int) * np.random.randint(-1,2, size=img_size)
    noise = np.array(np.random.normal(0, scale, img_size), dtype=np.int)
    noise = np.repeat(noise[:, :, np.newaxis], img.shape[2], axis=2)
    
    result = noise + 255*img  
    return np.clip(result,0,255)/255
  
def noise_batch(img_batch):
    result = []
    for i in range(len(img_batch)):
        tmp = noise(img_batch[i])
        result.append(tmp)
    return np.array(result)


def cifar10_train(model,Session, input_datas, input_labels, batch_size = 100, epochs = 10, print_accuracy = True, noise = False):
  
    sess = Session
    sess.run(tf.global_variables_initializer())
    
    train = model.training
    loss = model.loss
    
    total_batch = int(5*len(input_datas[0])/batch_size)

    for epoch in range(epochs):
        total_cost = 0
        idxs = list(range(len(input_datas[0])))
        np.random.shuffle(idxs)
        for i in range(0,len(input_datas[0]),batch_size):
            for step in range(5):
                input_data_ = input_datas[step]
                if noise:
                    input_data_ = noise_batch(input_data_)
                    
                input_labels_ = input_labels[step]
                _, cost_val = sess.run([train,loss], feed_dict = {model.input:input_data_[idxs[i:i+batch_size]], 
                                                                  model.labels:input_labels_[idxs[i:i+batch_size]]})
                total_cost = cost_val/total_batch

        print('Epoch: {}'.format(epoch+1), 'Avg_cost: {}'.format(total_cost))
        
        if print_accuracy:
            
            is_correct = tf.equal(tf.argmax(model.outputs,axis = 1), tf.argmax(model.labels_one_hot,axis = 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            print('Accuracy: ', sess.run(accuracy, feed_dict = {model.input: input_datas[5],
                                                                model.labels: input_labels[5]} ))
    print('Done')

