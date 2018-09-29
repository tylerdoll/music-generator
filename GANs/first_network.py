import tensorflow as tf
import keras
import numpy as np
from numpy import array
from data_generator import chord_data_set

print ("program start -----------------------------------------------------------------------------------------------")

input_layer = keras.layers.Input((24,))
x = keras.layers.Dense(24, activation=tf.nn.sigmoid)(input_layer)
x = keras.layers.Dense(2, activation=tf.nn.softmax)(x)

model = keras.Model(input_layer, x)

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
			  
			  
input_train, labels_train = chord_data_set(12800)
input_test, labels_test = chord_data_set(1000)

print ("labels_train[0]")
print (labels_train[0])
print (array(labels_train).shape)

model.fit(input_train, labels_train, epochs=50)

loss, accuracy = model.evaluate(input_test, labels_test)

print ("test accuracy: " + str(accuracy))
print ("test loss: " + str(loss))

predictions = model.predict(input_test)

for i in range(10):
	print ("i = " + str(i))
	print (input_test[i])
	print (predictions[i])
	print (labels_test[i])




