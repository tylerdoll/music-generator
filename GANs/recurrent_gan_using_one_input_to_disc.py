import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import keras
from numpy import array
#from data_generator import chord_data_set
from tensorflow.python.keras.layers import Input, Dense

"""
  The fake chord data generator
"""
import random
from random import randint
from numpy import array
import numpy
NOTE_SPACE = 6      # notes valid here.

RANDOMIZATION_MAX = 0.0

def get_chromatic_note_space(history_length, always_true=False):
	answer = numpy.random.uniform(low=0.0, high=RANDOMIZATION_MAX, size=(NOTE_SPACE * history_length,))
	base = randint(0, NOTE_SPACE)
	
	
	for i in range(history_length):
		start_of_this_frame = NOTE_SPACE * i
		
		offset = base + i
		#time_periods = get_chromatic_note_space[i]
		answer[start_of_this_frame + offset % NOTE_SPACE] = numpy.random.uniform(low=(1.0-RANDOMIZATION_MAX))

	if always_true or random.uniform(0, 1) < 0.5:
		
  
		# make a new pattern of just one note space size with the next note of the chromatic scale inside it
		expected_generation = numpy.random.uniform(low=0.0, high=RANDOMIZATION_MAX, size=(NOTE_SPACE,))
		expected_generation[(base + history_length) % NOTE_SPACE] = numpy.random.uniform(low=(1.0-RANDOMIZATION_MAX))
	
		label = 0
		# print (answer)
		# print (expected_generation)
		
	else:
		#for i in range(history_length):
		#	note = randint(0, NOTE_SPACE)
		#	answer[note] = numpy.random.uniform(low=(1.0-RANDOMIZATION_MAX))
		label = 1
		expected_generation = numpy.random.uniform(low=0.0, high=RANDOMIZATION_MAX, size=(NOTE_SPACE,))
		true_val = -1
		while true_val == -1 or true_val == (base + history_length) % NOTE_SPACE:
			true_val = randint(0, NOTE_SPACE - 1)
			
		#add a wrong note
		expected_generation[true_val] = numpy.random.uniform(low=(1.0-RANDOMIZATION_MAX))
		
		
	return numpy.concatenate([answer, expected_generation]), array(label)#, array(expected_generation)
	
	
def chromatic_data_set(size, history_length, always_true=False):
	data = []
	labels = []
	#expected_values = []
	for i in range(size):
		#time_periods = [None] * history_length
		
		input, output = get_chromatic_note_space(history_length, always_true)

		data.append(input)
		labels.append(output)
		#expected_values.append(expected_generation)
		
	return array(data), array(labels)#, array(expected_values)

# Here's some GAN code I got working, lets see what happens

#https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
#is helpful
#https://github.com/tensorflow/models/tree/master/research/gan

#https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py


import sys

print("running in version: " + sys.version)

import tensorflow as tf
import keras
import numpy as np
from numpy import array
#from data_generator import chord_data_set


print ("program start -----------------------------------------------------------------------------------------------")
#generator
NOISE_SIZE = 11
GAUSSIAN_STDDEV = 0.6

HISTORY_LENGTH = 1
#JOINING_LAYER_SIZE = HISTORY_LENGTH * NOTE_SPACE


TOTAL_HISTORY_SIZE = HISTORY_LENGTH * NOTE_SPACE
#generator_noise_input = keras.layers.Input((NOISE_SIZE,))
#history_input = keras.layers.Input((HISTORY_LENGTH * NOTE_SPACE,))
generator_input = keras.layers.Input((TOTAL_HISTORY_SIZE + NOISE_SIZE,))
# Used to combine the random noise vector and the history to generate from
# g = keras.layers.concatenate([history_input, generator_noise_input])

import keras.backend as K
# t = K.ones((15,))
# t1 = t[:3]
# t2 = t[1:]
# t3 = K.concatenate([t1, t2])
# print(K.eval(t3))

def get_history(x):
	return keras.backend.slice(x, (0, 0), (0, TOTAL_HISTORY_SIZE))
	#return x[:TOTAL_HISTORY_SIZE]
	#return tf.slice(x, [,0], [TOTAL_HISTORY_SIZE,], "custom_slice")
	# print ("x shape = " + str(tf.shape(x)))
	# split0, split1 = tf.split(x, [TOTAL_HISTORY_SIZE, NOISE_SIZE], 0)
	# print ("shapes")
	# print (tf.shape(split0))
	# print (tf.shape(split1))
	
#exit()
#hist_placeholder = tf.placeholder(shape=(TOTAL_HISTORY_SIZE, ), dtype=tf.float64)
print((TOTAL_HISTORY_SIZE + NOISE_SIZE,))
print ("gen_input is of shpae: " + str(tf.shape(generator_input)))
history_storage = keras.layers.Lambda(get_history, output_shape=(TOTAL_HISTORY_SIZE,))(generator_input)
#print (K.eval(history_storage))

g = keras.layers.Dense(NOTE_SPACE)(generator_input)
g = keras.layers.LeakyReLU(alpha=0.2)(g)

#g = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(g)
g = keras.layers.Dense(NOTE_SPACE)(g) # this matches up with the input layer of the discriminator
g = keras.layers.LeakyReLU(alpha=0.2)(g)

g = keras.layers.Dense(NOTE_SPACE, activation="sigmoid")(g)			#doing this to keep values in range (0, 1)

generator_output = keras.layers.concatenate([history_storage, g])
#print (K.eval(generator_output))
#g = keras.layers.BatchNormalization(momentum=0.8)(g)
#g = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(g)


#discriminator

#discriminator_history_input = keras.layers.Input((NOTE_SPACE * HISTORY_LENGTH,))

discriminator_input = keras.layers.Input(((HISTORY_LENGTH + 1) * NOTE_SPACE,))		   #this matches up with the output layer of the generator
#x = LSTM(512)(discriminator_input)
#x = LSTM(512)(x)

# Inputs
#x = keras.layers.concatenate([history_input, discriminator_next_time_period_input])
# put gaussian noise here to make the discriminator worse if it is too good for the generator


x = keras.layers.Dense(NOTE_SPACE * (HISTORY_LENGTH + 1))(discriminator_input)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dense(2, activation=tf.nn.softmax)(x)


generator = keras.Model(inputs=generator_input, outputs = generator_output)
print ("---------------------------------  Generator -----------------------------")
for layer in generator.layers:
    print(layer.get_output_at(0).get_shape().as_list())
discriminator = keras.Model(inputs=discriminator_input, outputs=x)
#generator = keras.Model(inputs=history_input, outputs = g)

# def set_discriminator_trainable(new_value):
   # # for l in discriminator.layers:
	# #	 l.trainable = new_value
	# discriminator.trainable = new_value
	# discriminator.compile(optimizer=tf.train.AdamOptimizer(), 
				  # loss='sparse_categorical_crossentropy',
				  # metrics=['accuracy'])
	# combined_system.compile(optimizer=tf.train.AdamOptimizer(), 
							# loss='sparse_categorical_crossentropy',
							# metrics=['accuracy'])

discriminator.compile(optimizer=keras.optimizers.Adam(), 
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])
			  
discriminator.trainable = False
			
			
#combined_system = keras.Model(inputs=[history_input, generator_noise_input], outputs=generator_output)
#merge_layer = keras.layers.concatenate()
combined_system = keras.Sequential()
#combined_system.add(keras.layers.concatenate())
combined_system.add(generator)
#combined_system.add(discriminator)
combined_system.compile(optimizer=tf.train.AdamOptimizer(), 
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
		


##########################################################################################################################
#			Discriminator testing
##########################################################################################################################

#input_train, labels_train = chromatic_data_set(TRAINING_INPUT_SIZE, HISTORY_LENGTH, always_true=True)

# TRAINING_INPUT_SIZE = 12800
# input_train, labels_train = chromatic_data_set(TRAINING_INPUT_SIZE, HISTORY_LENGTH)
# input_test, labels_test = chromatic_data_set(1000, HISTORY_LENGTH)


# discriminator.fit(input_train, labels_train, epochs=50)#50 seems like a good number, 20 for faster testing

# loss, accuracy = discriminator.evaluate(input_test, labels_test)

# print ("test accuracy: " + str(accuracy))
# print ("test loss: " + str(loss))

# predictions = discriminator.predict(input_test)

# for i in range(10):
	# print ("i = " + str(i))
	# print (input_test[i])
	# #print (generated_test[i])
	# print (predictions[i])
	# print (labels_test[i])
	
# exit()

##########################################################################################################################
#			END Discriminator testing
##########################################################################################################################


#noise_test = np.random.uniform(-1.0, 1.0, size=[1000, NOISE_SIZE])
#full_system_expected_out = np.ones([1000, 1])


	
#after making the discriminator, try to add in the generator

BATCH_COUNT = 400
#DISCRIMINATOR_PRETRAINING_BATCHES = 5
TRAINING_INPUT_SIZE = 256
TESTING_INPUT_SIZE = 1000


for batch in range(BATCH_COUNT):
	print ("batch counter = " + str(batch))

	# Discriminator training
	random_inputs_train = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
	#random_inputs_test = np.random.uniform(-1.0, 1.0, size=[TESTING_INPUT_SIZE, NOISE_SIZE])
		
	input_train, labels_train = chromatic_data_set(TRAINING_INPUT_SIZE, HISTORY_LENGTH, always_true=True)
	#input_test, labels_test = chromatic_data_set(TESTING_INPUT_SIZE, HISTORY_LENGTH, always_true=True)

	# make the generator input
	generator_input = []
	for i in range(TRAINING_INPUT_SIZE):
		generator_input.append(numpy.concatenate([input_train[i][:TOTAL_HISTORY_SIZE], random_inputs_train[i]]))
	generator_input = array(generator_input)
	
	generated_patterns_train = generator.predict(generator_input)
	#generated_patterns_test = generator.predict(random_inputs_test)
	continue
	all_inputs = np.concatenate((generated_patterns_train, input_train))
	all_labels = np.concatenate((np.ones([TRAINING_INPUT_SIZE, 1]), np.zeros([TRAINING_INPUT_SIZE, 1])))
	
	#all_inputs_test = np.concatenate((generated_patterns_test, input_test))
	#all_labels_test = np.concatenate((np.ones([TESTING_INPUT_SIZE, 1]), np.zeros([TESTING_INPUT_SIZE, 1])))

	#discriminator.fit(input_train, labels_train, epochs=1)
	discriminator.fit(all_inputs, all_labels, epochs=1)
	
	#evaluating discriminator accuracy:
	
	# loss, accuracy = discriminator.evaluate(all_inputs_test, all_labels_test)
	# print ("discriminator test accuracy: " + str(accuracy))
	# print ("discriminator test loss: " + str(loss))
	
	#if batch < DISCRIMINATOR_PRETRAINING_BATCHES:	#use this to pretrain the discriminator without duplicating code
	#	continue
	#generator training, by training batches through the entire model.
	#set_discriminator_trainable(False)
	#random_inputs = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
	
	desired_outputs_train = np.zeros([TRAINING_INPUT_SIZE, 1])
	#desired_outputs_test = np.zeros([TESTING_INPUT_SIZE, 1])
	combined_system.fit(random_inputs_train, desired_outputs_train, epochs=1)
	
	#combined_system.train_on_batch(random_inputs, desired_outputs)
	#evaluating generator accuracy:
	#loss, accuracy = combined_system.evaluate(noise_test, full_system_expected_out)
	# loss, accuracy = combined_system.evaluate(random_inputs_test, desired_outputs_test)
	# print ("generator fooled discriminator rate: " + str(accuracy))
	# print ("combined_system test loss: " + str(loss))
	

testing_random_inputs = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
generated_patterns = generator.predict(testing_random_inputs)
discriminated_patterns = discriminator.predict(generated_patterns)

for i in range(10):
	print ("i = " + str(i))
	#print (input_test[i])
	print (generated_patterns[i])
	print (discriminated_patterns[i])
	#print (labels_test[i])
	
