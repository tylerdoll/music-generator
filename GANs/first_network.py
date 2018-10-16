#!C:\Program Files\Python36\python.exe

#https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
#is helpful

import sys

print("running in version: " + sys.version)

import tensorflow as tf
import keras
import numpy as np
from numpy import array
from data_generator import chord_data_set

print ("program start -----------------------------------------------------------------------------------------------")
#generator
NOISE_SIZE = 128
generator_input = keras.layers.Input((NOISE_SIZE,))
g = keras.layers.Dense(24, activation=tf.nn.sigmoid)(generator_input)
g = keras.layers.Dense(24, activation=tf.nn.sigmoid)(g) #this matches up with the input layer of the discriminator
#discriminator
discriminator_input = keras.layers.Input((24,))        #this matches up with the output layer of the generator
x = keras.layers.Dense(24, activation=tf.nn.sigmoid)(discriminator_input)
x = keras.layers.Dense(2, activation=tf.nn.softmax)(x)


discriminator = keras.Model(discriminator_input, x)
generator = keras.Model(generator_input, g)

def set_discriminator_trainable(new_value):
   # for l in discriminator.layers:
    #    l.trainable = new_value
    discriminator.trainable = new_value
    discriminator.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    combined_system.compile(optimizer=tf.train.AdamOptimizer(), 
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

discriminator.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
			  
			
combined_system = keras.Sequential()
combined_system.add(generator)
combined_system.add(discriminator)
combined_system.compile(optimizer=tf.train.AdamOptimizer(), 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            
TRAINING_INPUT_SIZE = 12800
input_train, labels_train = chord_data_set(TRAINING_INPUT_SIZE)
input_test, labels_test = chord_data_set(1000)

print ("labels_train[0]")
print (labels_train[0])
print (array(labels_train).shape)

discriminator.fit(input_train, labels_train, epochs=20)#50 seems like a good number, 20 for faster testing

loss, accuracy = discriminator.evaluate(input_test, labels_test)

print ("test accuracy: " + str(accuracy))
print ("test loss: " + str(loss))

predictions = discriminator.predict(input_test)

noise_test = np.random.uniform(-1.0, 1.0, size=[1000, NOISE_SIZE])
full_system_expected_out = np.ones([1000, 1])

for i in range(10):
	print ("i = " + str(i))
	print (input_test[i])
	print (predictions[i])
	print (labels_test[i])
    
#after making the discriminator, try to add in the generator
BATCH_COUNT = 100
set_discriminator_trainable(False)
for batch in range(BATCH_COUNT):
    print ("batch counter = " + str(batch))
    #discriminator training - needed to keep it working correctly
    # set_discriminator_trainable(True)
    # random_inputs = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
    # generated_patterns = generator.predict(random_inputs)
    # input_train, labels_train = chord_data_set(TRAINING_INPUT_SIZE, all_major=True)
    # all_inputs = np.concatenate((generated_patterns, input_train))
    # expected_discriminator_outputs = np.concatenate((np.ones([TRAINING_INPUT_SIZE, 1]), np.zeros([TRAINING_INPUT_SIZE, 1])))
    # discriminator.train_on_batch(all_inputs, expected_discriminator_outputs)
    
    #evaluating discriminator accuracy:
    # loss, accuracy = discriminator.evaluate(input_test, labels_test)
    # print ("discriminator test accuracy: " + str(accuracy))
    # print ("discriminator test loss: " + str(loss))
    
    #generator training, by training batches through the entire model.
    #set_discriminator_trainable(False)
    random_inputs = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
    desired_outputs = np.zeros([TRAINING_INPUT_SIZE, 1])
    combined_system.train_on_batch(random_inputs, desired_outputs)
    #evaluating generator accuracy:
    #loss, accuracy = combined_system.evaluate(noise_test, full_system_expected_out)
    loss, accuracy = combined_system.evaluate(random_inputs, desired_outputs)
    print ("combined_system test accuracy: " + str(accuracy))
    print ("combined_system test loss: " + str(loss))
    

testing_random_inputs = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
generated_patterns = generator.predict(testing_random_inputs)
discriminated_patterns = discriminator.predict(generated_patterns)

for i in range(10):
    print ("i = " + str(i))
    #print (input_test[i])
    print (generated_patterns[i])
    print (discriminated_patterns[i])
    # print (labels_test[i])
        
    
    
    
    
    
    
    
    
    