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
NOTE_SPACE = 5      #5 octaves of notes are valid here.

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
		
		
	return array(answer), array(label), array(expected_generation)
	
	
def chromatic_data_set(size, history_length, always_true=False):
	data = []
	labels = []
	expected_values = []
	for i in range(size):
		#time_periods = [None] * history_length
		
		input, output, expected_generation = get_chromatic_note_space(history_length, always_true)

		data.append(input)
		labels.append(output)
		expected_values.append(expected_generation)
		
	return array(data), array(labels), array(expected_values)

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
NOISE_SIZE = 128
GAUSSIAN_STDDEV = 0.6

HISTORY_LENGTH = 5
#JOINING_LAYER_SIZE = HISTORY_LENGTH * NOTE_SPACE
JOINING_LAYER_SIZE = NOTE_SPACE

generator_noise_input = keras.layers.Input((NOISE_SIZE,))
history_input = keras.layers.Input((HISTORY_LENGTH * NOTE_SPACE,))
# Used to combine the random noise vector and the history to generate from
g = keras.layers.concatenate([history_input, generator_noise_input])

g = keras.layers.Dense(NOTE_SPACE)(g)
g = keras.layers.LeakyReLU(alpha=0.2)(g)

#g = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(g)
g = keras.layers.Dense(NOTE_SPACE)(g) # this matches up with the input layer of the discriminator
g = keras.layers.LeakyReLU(alpha=0.2)(g)

generator_output = keras.layers.Dense(JOINING_LAYER_SIZE, activation="sigmoid")(g)			#doing this to keep values in range (0, 1)
#g = keras.layers.BatchNormalization(momentum=0.8)(g)
#g = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(g)


#discriminator

#discriminator_history_input = keras.layers.Input((NOTE_SPACE * HISTORY_LENGTH,))

discriminator_next_time_period_input = keras.layers.Input((NOTE_SPACE,))		   #this matches up with the output layer of the generator
#x = LSTM(512)(discriminator_input)
#x = LSTM(512)(x)

# Inputs
x = keras.layers.concatenate([history_input, discriminator_next_time_period_input])
# put gaussian noise here to make the discriminator worse if it is too good for the generator


x = keras.layers.Dense(NOTE_SPACE * (HISTORY_LENGTH + 1))(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dense(2, activation=tf.nn.softmax)(x)


discriminator = keras.Model(inputs=[history_input, discriminator_next_time_period_input], outputs=x)
generator = keras.Model(inputs=[history_input, generator_noise_input], outputs = generator_output)
#generator = keras.Model(inputs=history_input, outputs = g)

discriminator.compile(optimizer=keras.optimizers.Adam(), 
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])
			  
discriminator.trainable = False
			

#input_history = keras.Sequential()
#input_history.add(keras.layers.GaussianNoise(0.0))

#disc_input_next_frame = keras.Sequential()
#disc_input_next_frame.add(keras.layers.GaussianNoise(0.0))
	
#disc_input_merged = keras.layers.Concatenate([history_input, generator])

#gen_input_noise = keras.Sequential()

#gen_input_merged = keras.layers.Concatenate([history_input, generator_noise_input])
#gen_input_noise.add(keras.layers.GaussianNoise)
#combined_system = keras.Model(inputs=[history_input, generator_noise_input], outputs=generator_output)

#merged_inputs = keras.layers.concatenate([history_input, generator_noise_input])
#disc_translation = keras.layers.concatenate([history_input, generator])
#disc = 
combined_system = keras.Model(input=generator.input, output=discriminator([history_input, generator.output]))
# combined_system = keras.Sequential()
# combined_system.add(gen_input_merged)
# combined_system.add(generator)

# combined_system.add(disc_input_merged)
# combined_system.add(discriminator)
combined_system.compile(optimizer=tf.train.AdamOptimizer(), 
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
		


##########################################################################################################################
#			Discriminator testing
##########################################################################################################################

# TRAINING_INPUT_SIZE = 12800
# input_train, labels_train, generated_train = chromatic_data_set(TRAINING_INPUT_SIZE, HISTORY_LENGTH)
# input_test, labels_test, generated_test = chromatic_data_set(1000, HISTORY_LENGTH)


# discriminator.fit([input_train, generated_train], labels_train, epochs=50)#50 seems like a good number, 20 for faster testing

# loss, accuracy = discriminator.evaluate([input_test, generated_test], labels_test)

# print ("test accuracy: " + str(accuracy))
# print ("test loss: " + str(loss))

# predictions = discriminator.predict([input_test, generated_test])

# for i in range(10):
	# print ("i = " + str(i))
	# print (input_test[i])
	# print (generated_test[i])
	# print (predictions[i])
	# print (labels_test[i])
	
# exit()

##########################################################################################################################
#			END Discriminator testing
##########################################################################################################################


#noise_test = np.random.uniform(-1.0, 1.0, size=[1000, NOISE_SIZE])
#full_system_expected_out = np.ones([1000, 1])


	
#after making the discriminator, try to add in the generator

BATCH_COUNT = 200
#DISCRIMINATOR_PRETRAINING_BATCHES = 5
TRAINING_INPUT_SIZE = 12800
TESTING_INPUT_SIZE = 1000


for batch in range(BATCH_COUNT):
	print ("batch counter = " + str(batch))

	# Discriminator training
	random_inputs_train = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
	#random_inputs_test = np.random.uniform(-1.0, 1.0, size=[TESTING_INPUT_SIZE, NOISE_SIZE])
	
	history, labels, expected_generation = chromatic_data_set(TRAINING_INPUT_SIZE, HISTORY_LENGTH, always_true=True)
	#input_test, labels_test = chromatic_data_set(TESTING_INPUT_SIZE, HISTORY_LENGTH, always_true=True)

	generated_patterns = generator.predict([history, random_inputs_train])
	#generated_patterns_test = generator.predict(random_inputs_test)	
	
	all_history = np.concatenate((history, history))
	all_labels = np.concatenate((np.ones([TRAINING_INPUT_SIZE, 1]), np.zeros([TRAINING_INPUT_SIZE, 1])))
	all_expected_generations = np.concatenate((generated_patterns, expected_generation))

	#all_inputs_test = np.concatenate((generated_patterns_test, input_test))
	#all_labels_test = np.concatenate((np.ones([TESTING_INPUT_SIZE, 1]), np.zeros([TESTING_INPUT_SIZE, 1])))

	#discriminator.fit(input_train, labels_train, epochs=1)
	discriminator.fit([all_history, all_expected_generations], all_labels, epochs=1)
	
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
	combined_system.fit([history, random_inputs_train], desired_outputs_train, epochs=1)
	
	#combined_system.train_on_batch(random_inputs, desired_outputs)
	#evaluating generator accuracy:
	#loss, accuracy = combined_system.evaluate(noise_test, full_system_expected_out)
	# loss, accuracy = combined_system.evaluate(random_inputs_test, desired_outputs_test)
	# print ("generator fooled discriminator rate: " + str(accuracy))
	# print ("combined_system test loss: " + str(loss))
	

testing_random_inputs = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
history_test, labels_test, expected_generation_test = chromatic_data_set(TRAINING_INPUT_SIZE, HISTORY_LENGTH, always_true=True)
generated_patterns = generator.predict([history, testing_random_inputs])
discriminated_patterns = discriminator.predict([history_test, generated_patterns])

for i in range(10):
	print ("i = " + str(i))
	#print (input_test[i])
	print (history_test[i])
	print (generated_patterns[i])
	print (discriminated_patterns[i])
	#print (labels_test[i])
	