import tensorflow as tf 
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.utils import to_categorical


cifar10 =keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#model = tf.keras.models.Sequential()
model = Sequential()
#model.add(vgg_model.layers[0])
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,3)))
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, shuffle=True, validation_data =(x_test, y_test), epochs=1)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
