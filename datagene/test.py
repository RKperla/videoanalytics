import numpy as np
import pdb
from keras.models import Sequential
from DataGenerator import DataGenerator

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 1,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

pdb.set_trace()
# Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
#model = Sequential()
#[...] # Architecture
#model.compile()

# Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)