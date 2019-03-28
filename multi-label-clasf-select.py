
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
import pdb
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

K.set_learning_phase(1) #set learning phase

from keras.models import Sequential
"""Import from keras_preprocessing not from keras.preprocessing, because Keras may or maynot contain the features discussed here depending upon when you read this article, until the keras_preprocessed library is updated in Keras use the github version."""
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

print 'loading dataframe'
df = pd.read_pickle("./upper_body_atts.pkl")
#df= df[0:10000]
df = df[['img_name','landmarks','atts']]

df["labels"] = df["atts"].apply(lambda x:list(x))

def name(x):
    name = str.split(x,'/')
    return '/fashion-attribute/rk/img_bbox_data/img_n/'+ name[1]+'/'  + name[2]
    #return '/fashion-attribute/storage_bucket/deep_fashion/img_bbox_data/img_n/' + name[1] + '/' + name[2]
#/fashion-attribute/storage_bucket/small_data_set/work/train/Blazer/Double-Breasted_Blazer

# local path : /Users/1024329/Downloads/DeepFashion/img_n/
# server path : /fashion-attribute/storage_bucket/deep_fashion/img_bbox_data/img_n/
    
df['Filenames'] = df['img_name'].apply(lambda x:name(x))

#df['Filenames'] = df['img_name'].apply(lambda x: '/Users/1024329/Downloads/DeepFashion/' + x)

# removing the samples without any attribute present in it
sample = df['atts'].apply(lambda x: x.shape[0]>0)
df4 =df[sample]


# In[2]:


df_cat = pd.read_csv('list_attr_cloth.csv')



df4 = df4.reset_index(drop =True)


# In[5]:


vals = np.array([])
for i in range(df4.shape[0]):
    if i%10000 ==0: print i
    vals = np.concatenate([vals,df4.loc[i,'labels']]) 


# In[6]:


y = np.bincount(vals.astype(int))
ii = np.nonzero(y)[0]
s = zip(ii,y[ii])
sa = [ i[0] for i in s if i[1] > 1000]


# In[7]:


print len(sa)


# In[8]:


s5= [44, 45, 99, 141, 162, 237, 438, 577, 653, 705, 760, 781, 823, 851, 892, 988]


# In[9]:


# removing the style category attributes
# and retaining texture, fabric, shape and part (style last one removed)
sa = [item for item in sa if item not in s5]


print np.array(sa)


# In[28]:


df4['catatts'] = df4['labels'].apply(lambda x: [item for item in sa if item in x])


# In[29]:


sample = df4['catatts'].apply(lambda x: len(x)>0)
df5 =df4[sample]


# In[30]:


df5 = df5.reset_index(drop= True)


# In[33]:


print df5.shape


# In[34]:



print 'creating image generators'
#import pdb
#pdb.set_trace()
img_reso = 224
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(dataframe=df5[0:70000],directory="", 
        x_col="Filenames",y_col="catatts",batch_size=32,
        seed=42,shuffle=True,class_mode="categorical",
        classes=list(sa),target_size=(img_reso,img_reso))

print 'validation generator'
valid_generator=test_datagen.flow_from_dataframe(
        dataframe=df5[70000:100000],
directory="",
x_col="Filenames",
y_col="catatts",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
classes=list(sa),
target_size=(img_reso,img_reso))

print 'test generator'
test_generator=test_datagen.flow_from_dataframe(dataframe=df5[100000:],
directory="",
x_col="Filenames",
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(img_reso,img_reso))


# In[39]:



network ='resnet50'

if network == 'vgg':
	from keras_frcnn import vgg as nn
elif network == 'resnet50':
	from keras_frcnn import resnet as nn
else:
	print('Not a valid model')
	raise ValueError


if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (224,224, 3)

img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=False)

out = (Flatten(name='flatten'))(shared_layers)
out = (Dense(4096, activation='relu', name='fc5'))(out)
out = (Dropout(0.5))(out)
out = (Dense(2048, activation='relu', name='fc6'))(out)
out = (Dropout(0.5))(out)
out = (Dense(512, activation='relu', name='fc7'))(out)
out_class = (Dense(86, activation='sigmoid'))(out)

model_classifier = Model(img_input, out_class)

try:
    print('loading weights from pretrained model')
    model_classifier.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder 		https://github.com/fchollet/keras/tree/master/keras/applications')

#model_classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])
model_classifier.compile(optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])



# checkpoint = ModelCheckpoint("cat1_atts.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)


# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
# model_classifier.fit_generator(generator=train_generator,
#                    steps_per_epoch=STEP_SIZE_TRAIN,
#                    validation_data=valid_generator,
#                    validation_steps=STEP_SIZE_VALID,
#                    epochs=50, use_multiprocessing=True, callbacks = [checkpoint])




# In[ ]:



checkpoint = ModelCheckpoint("deepf_cat4_atts86.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model_classifier.fit_generator(generator=train_generator,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_generator,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=50, use_multiprocessing=True, callbacks = [checkpoint])




# In[18]:


# model_classifier.load_weights('vgg16_atts.h5')
model_classifier.load_weights('cat1_atts.h5')


# In[19]:


# import keras
# model_f = keras.models.load_model('vgg16_atts.h5')


# In[20]:


a, b = train_generator.next()


# In[21]:


a.shape, b.shape


# In[22]:


out = model_classifier.predict(a)


# In[23]:


out.shape


# In[42]:


np.max(out,axis =1)


# In[50]:


num = 11
np.nonzero(out[num,:]> 0.1), np.nonzero(b[num,:] > 0.4)

