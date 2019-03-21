
# coding: utf-8

# In[1]:


from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import cv2
import pandas as pd

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn.mylayer import mylayer
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

K.set_learning_phase(1) #set learning phase

# sys.setrecursionlimit(40000)


# In[2]:


network ='vgg'

if network == 'vgg':
	from keras_frcnn import vgg as nn
elif network == 'resnet50':
	from keras_frcnn import resnet as nn
else:
	print('Not a valid model')
	raise ValueError


# In[3]:


from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (224,224, 3)

img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)


out = (Flatten(name='flatten'))(shared_layers)
out = (Dense(4096, activation='relu', name='fc5'))(out)
out = (Dropout(0.5))(out)
out = (Dense(2048, activation='relu', name='fc6'))(out)
out = (Dropout(0.5))(out)
out = (Dense(512, activation='relu', name='fc7'))(out)
out_class = (Dense(46, activation='sigmoid'))(out)

model_classifier = Model(img_input, out_class)


try:
	print('loading weights from pretrained model')
	model_classifier.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder 		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_classifier.compile(optimizer=optimizer_classifier, loss = 'binary_crossentropy', metrics= ['accuracy'])



def name(x):
    name = str.split(x,'/')
    return '/Users/1024329/Downloads/DeepFashion/img_n/' + name[1] + '/' + name[2]
    
df3 = pd.read_pickle('upper_body_atts.pkl')
df3['img_name'] = df3['img_name'].apply(lambda x:name(x))


# In[5]:

df3 = df3[['img_name','landmarks','atts']]

# In[6]:


# removing the samples without any attribute present in it
sample = df3['atts'].apply(lambda x: x.shape[0]>0)
df4 =df3[sample]
# checking the maximum value of the attribute label
df4['atts'].apply(lambda x: np.max(x)).max()

# reset index of df4
df4 = df4.reset_index(drop = True)


vals = np.array([])
for i in range(df4.shape[0]):
    if i%10000 ==0: print i
    vals = np.concatenate([vals,df4.loc[i,'atts']]) 


# In[9]:


import matplotlib.pyplot as plt
his = plt.hist(vals,bins =1000)
select =  np.nonzero(his[0] > 2000)


def selecting(x,select):
    s = [i for i in x if i in (select[0])]
    return s


# In[12]:


df4['select46_atts'] = df4['atts'].apply(lambda x: selecting(x,select))


# removing the samples without any attribute present in it
sample = df4['select46_atts'].apply(lambda x: np.array(x).shape[0]>0)
df5 =df4[sample]
print df5.shape, df4.shape
# checking the maximum value of the attribute label
print df5['select46_atts'].apply(lambda x: np.max(x)).max()

# reset index of df4
df5 = df5.reset_index(drop = True)


df5['select46_atts'] = df5['select46_atts'].apply(lambda x: np.array(x))


def get_batch(num):
    img = []
    attribs = []
    b_size = 2
    for i in range(b_size*num, b_size*(num+1)):
        #print i
        a = cv2.imread(df5.loc[i,'img_name'])
        a = cv2.resize(a,(224,224))
        a = np.array(a).reshape(224,224,3)
        img.append(a/255)
        atts = df5.loc[i,'select46_atts']
        labels = np.array([0 if item not in atts else 1 for item in select[0]]).reshape(46,)
        attribs.append(labels)
    return np.array(img), np.array(attribs)



df5 = df5.head(35000)


# In[21]:


df5 = df5.sample(frac = 1).reset_index(drop =True)


# In[ ]:


train_loss = []
# train_acc = []
for j in range(50):
    df5 = df5.sample(frac = 1).reset_index(drop =True)
    print 'epoch number:', j
    los = 0
    acc = 0
    for i in range(1000):
        print 'epoch and batch:', j, i 
        X,Y = get_batch(i)
        history = model_classifier.fit(x=X, y= Y, verbose = 2)
        los = los + np.array(history.history['loss'])
        #acc = acc + np.array(history.history['acc'])
        #print los
    train_loss.append(los)
    model_classifier.save_weights('atts_vgg_model.h5')
    #train_acc.append(acc)
    #print train_loss, train_acc
        
