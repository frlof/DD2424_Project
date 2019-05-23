#https://engmrk.com/alexnet-implementation-using-keras/

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)

import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers
from keras.callbacks import LearningRateScheduler

from keras import optimizers
'''
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
'''
'''
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate
'''
weight_decay = 1e-4
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
#model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
#model.add(BatchNormalization())

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))

# 2nd Convolutional Layer
#model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
#model.add(BatchNormalization())

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))

# 3rd Convolutional Layer
#model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 4th Convolutional Layer
#model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 5th Convolutional Layer
#model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
#model.add(BatchNormalization())

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
#model.add(Dense(512, input_shape=(32*32*3,), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(512, input_shape=(32*32*3,)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Add Dropout to prevent overfitting
#model.add(Dropout(0.5))

# 2nd Fully Connected Layer
#model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Add Dropout
#model.add(Dropout(0.5))

# 3rd Fully Connected Layer
#model.add(Dense(250, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(BatchNormalization())

#model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# Compile the model
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=["accuracy"]) 

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt_rms, metrics=["accuracy"]) 

#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 


cifar10 =keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train = x_train.reshape(50000,32,32,3)
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)



plotdata = model.fit(x_train, y_train, batch_size=100, shuffle=True, validation_data =(x_test, y_test), epochs=50)

#plotdata = model.fit_generator(datagen.flow(x_train, y_train, shuffle=True, batch_size=100), steps_per_epoch=len(x_train)/100, epochs=50, validation_data =(x_test, y_test))

#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
#                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
#                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

plt.plot(plotdata.history['acc'])
plt.plot(plotdata.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(plotdata.history['loss'])
plt.plot(plotdata.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()