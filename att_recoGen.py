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

from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

K.set_learning_phase(1) #set learning phase

network ='vgg'

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
	input_shape_img = (400, 400, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(6, 4))
nb_classes = 575

#pdb.set_trace()
num_rois = 6
# compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

if K.backend() == 'tensorflow':
    pooling_regions = 4
    input_shape = (num_rois,4,4,512)
elif K.backend() == 'theano':
    pooling_regions = 7
    input_shape = (num_rois,512,7,7)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)


out_roi_pool = mylayer(pooling_regions, 6)([shared_layers, roi_input])
#print out_roi_pool

out = (Flatten(name='flatten'))(out_roi_pool)
out = (Dense(4096, activation='relu', name='fc5'))(out)
out = (Dropout(0.5))(out)
out = (Dense(2048, activation='relu', name='fc6'))(out)
out = (Dropout(0.5))(out)
print 'no of classes', nb_classes
out_class = (Dense(1000, activation='sigmoid', kernel_initializer='zero'))(out)

model_classifier = Model([img_input, roi_input], out_class)

model_w  = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


try:
	print('loading weights from pretrained model')
	#model_classifier.load_weights(model_w, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_classifier.compile(optimizer=optimizer_classifier, loss = 'binary_crossentropy',metrics= ['accuracy'])


class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes = None, target_size=(256, 256),
                 color_mode = 'rgb', classes=None, class_mode = 'categorical', batch_size = 32,
                 shuffle = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix = '', save_format = 'jpeg', follow_links = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes
 
    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['origin']['x'], bounding_box['origin']['y'], bounding_box['width'],
                     bounding_box['height']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y


train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()


dict_train = dict({'img_00000001.jpg': 2, 'img_00000002.jpg': 3,'img_00000003.jpg': 4 })

######### change the dir here 
pdb.set_trace()

train_iterator = DirectoryIteratorWithBoundingBoxes("/Users/1024329/deep_learning/code/data/train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200))

# test_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/val", test_datagen, bounding_boxes=dict_val,target_size=(200, 200))

# lr_reducer = ReduceLROnPlateau(monitor='val_loss',
#                                patience=12,
#                                factor=0.5,
#                                verbose=1)
# tensorboard = TensorBoard(log_dir='./logs')

# early_stopper = EarlyStopping(monitor='val_loss',
#                               patience=30,
#                               verbose=1)

# checkpoint = ModelCheckpoint('./models/model.h5')

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

model_classifier.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=2000,
                          epochs=200, validation_data=custom_generator(test_iterator),
                          validation_steps=200,
                          verbose=1,
                          workers=12)


# model_classifier.fit_generator(custom_generator(train_iterator),
#                           steps_per_epoch=2000,
#                           epochs=200, validation_data=custom_generator(test_iterator),
#                           validation_steps=200,
#                           verbose=2,
#                           shuffle=True,
#                           callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
#                           workers=12)


scores = model_classifier.evaluate_generator(custom_generator(test_iterator), steps=2000)

print('Multi target loss: ' + str(scores[0]))
print('Image loss: ' + str(scores[1]))
print('Bounding boxes loss: ' + str(scores[2]))
print('Image accuracy: ' + str(scores[3]))
print('Top-5 image accuracy: ' + str(scores[4]))
print('Bounding boxes error: ' + str(scores[5]))


# data loading
df3 = pd.read_pickle('upper_body_atts.pkl')
import pdb
pdb.set_trace()

def name(x):
    name = str.split(x,'/')
    return '/Users/1024329/Downloads/DeepFashion/img_n/' + name[1] + '/' + name[2]

# local path : /Users/1024329/Downloads/DeepFashion/img_n/
# server path : /fashion-attribute/storage_bucket/deep_fashion/img_bbox_data/img_n/
    
df3['img_name'] = df3['img_name'].apply(lambda x:name(x))

# removing the samples without any attribute present in it
sample = df3['atts'].apply(lambda x: x.shape[0]>0)
df4 =df3[sample]
df4 = df4.reset_index(drop = True)
#pdb.set_trace()
def get_batch(i):
    a = cv2.imread(df4.loc[i,'img_name'])
    a = cv2.resize(a,(400,400))        
    img = np.array(a).reshape(1,400,400,3)
    lnd = df4.loc[i,'landmarks']
    landmarks = (np.array(lnd)/8).astype(int).reshape(1,6,4)
    atts = df4.loc[i,'atts']
    labels = np.array([0 if k not in atts else 1 for k in range(1000)]).reshape(1,1000)
    return img, landmarks, labels

train_loss = []
train_acc = []
pdb.set_trace()
for j in range(20):
    print 'epoch number:', j
    los = 0
    acc = 0
    for i in range(50000):
        print 'epoch and batch number:', j, i
        X,X2,Y = get_batch(i)
        history = model_classifier.fit(x=[X,X2], y= Y, verbose = 0)
        los = los + np.array(history.history['loss'])
        acc = acc + np.array(history.history['acc'])
        if i %1000 ==0:
            print 'saving model', j, i
            model_classifier.save_weights('att1kmodel.h5')
    train_loss.append(los)
    train_acc.append(acc)
    model_classifier.save_weights('atts_model.h5')
    np.save('loss.csv',train_loss)
#pdb.set_trace()
