#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class vgg16(): 
    def __init__(self, input_shape=[32,32,3], num_classes=10, light_ver = True):
        
        self.light_ver = light_ver
        self.num_classes = num_classes
        
        self.logits = self.model(input_shape)
        self.outputs = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.labels_one_hot))
    
    def model(self, input_shape):
      
        self.input = tf.placeholder(tf.float32, shape = [None]+input_shape)
        self.labels = tf.placeholder(tf.int32, shape = [None])
        self.labels_one_hot = tf.one_hot(self.labels,self.num_classes)
        
        x = tf.layers.conv2d(self.input, 64, kernel_size=[3,3], padding = 'SAME',activation =tf.nn.relu)
        x = tf.layers.conv2d(x, 64, kernel_size = [3,3], padding = 'SAME', activation = tf.nn.relu) 
        x = tf.layers.max_pooling2d(x, [2,2],[2,2],padding = 'SAME')
        
        x = tf.layers.conv2d(x, 128, kernel_size=[3,3], padding = 'SAME',activation =tf.nn.relu)
        x = tf.layers.conv2d(x, 128, kernel_size = [3,3], padding = 'SAME', activation = tf.nn.relu) 
        x = tf.layers.max_pooling2d(x, [2,2],[2,2],padding = 'SAME') 
        
        x = tf.layers.conv2d(x, 256, kernel_size=[3,3], padding = 'SAME',activation =tf.nn.relu)
        x = tf.layers.conv2d(x, 256, kernel_size = [3,3], padding = 'SAME', activation = tf.nn.relu) 
        x = tf.layers.max_pooling2d(x, [2,2],[2,2],padding = 'SAME') 
        
        if self.light_ver == True:
            x = tf.layers.flatten(x) 
            x = tf.layers.dense(x, 256) 
            logit = tf.layers.dense(x, self.num_classes, activation = None) 
            return logit
        
        x = tf.layers.conv2d(x, 512, kernel_size=[3,3], padding = 'SAME',activation =tf.nn.relu)
        x = tf.layers.conv2d(x, 512, kernel_size = [3,3], padding = 'SAME', activation = tf.nn.relu) 
        x = tf.layers.max_pooling2d(x, [2,2],[2,2],padding = 'SAME')
        
        x = tf.layers.conv2d(x, 1024, kernel_size=[3,3], padding = 'SAME',activation =tf.nn.relu)
        x = tf.layers.conv2d(x, 1024, kernel_size = [3,3], padding = 'SAME', activation = tf.nn.relu) 
        x = tf.layers.max_pooling2d(x, [2,2],[2,2],padding = 'SAME') 
       
        x = tf.layers.flatten(x) 
        x = tf.layers.dense(x, 256) 
        logit = tf.layers.dense(x, self.num_classes, activation = None) 
        
        return logit
        
    def train(self, lr=1e-3, optimizer = tf.train.AdamOptimizer):
        self.optimizer = optimizer(learning_rate=lr)
        self.training = self.optimizer.minimize(self.loss)
        

