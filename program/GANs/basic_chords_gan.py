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
GAUSSIAN_STDDEV = 0.2

generator_input = keras.layers.Input((NOISE_SIZE,))
g = keras.layers.Dense(24)(generator_input)
g = keras.layers.LeakyReLU(alpha=0.2)(g)
#g = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(g)
g = keras.layers.Dense(24)(g) #this matches up with the input layer of the discriminator
g = keras.layers.LeakyReLU(alpha=0.2)(g)
g = keras.layers.Dense(24, activation="sigmoid")(g)			#doing this to keep values in range (0, 1)
#g = keras.layers.BatchNormalization(momentum=0.8)(g)
#g = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(g)
#discriminator
discriminator_input = keras.layers.Input((24,))		   #this matches up with the output layer of the generator
x = keras.layers.GaussianNoise(GAUSSIAN_STDDEV)(discriminator_input)
x = keras.layers.Dense(24)(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dense(2, activation=tf.nn.softmax)(x)


discriminator = keras.Model(discriminator_input, x)
generator = keras.Model(generator_input, g)

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
			
combined_system = keras.Sequential()
combined_system.add(generator)
combined_system.add(discriminator)
combined_system.compile(optimizer=tf.train.AdamOptimizer(), 
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
			
# TRAINING_INPUT_SIZE = 12800
# input_train, labels_train = chord_data_set(TRAINING_INPUT_SIZE)
# input_test, labels_test = chord_data_set(1000)

# print ("labels_train[0]")
# print (labels_train[0])
# print (array(labels_train).shape)

# discriminator.fit(input_train, labels_train, epochs=50)#50 seems like a good number, 20 for faster testing

# loss, accuracy = discriminator.evaluate(input_test, labels_test)

# print ("test accuracy: " + str(accuracy))
# print ("test loss: " + str(loss))

# predictions = discriminator.predict(input_test)

# for i in range(10):
	# print ("i = " + str(i))
	# print (input_test[i])
	# print (predictions[i])
	# print (labels_test[i])
	
# exit()

#noise_test = np.random.uniform(-1.0, 1.0, size=[1000, NOISE_SIZE])
#full_system_expected_out = np.ones([1000, 1])


	
#after making the discriminator, try to add in the generator

BATCH_COUNT = 2000
#DISCRIMINATOR_PRETRAINING_BATCHES = 5
TRAINING_INPUT_SIZE = 128
TESTING_INPUT_SIZE = 1000

#set_discriminator_trainable(False)
for batch in range(BATCH_COUNT):
	print ("batch counter = " + str(batch))
	#discriminator training - needed to keep it working correctly
	#set_discriminator_trainable(True)
	
	random_inputs_train = np.random.uniform(-1.0, 1.0, size=[TRAINING_INPUT_SIZE, NOISE_SIZE])
	random_inputs_test = np.random.uniform(-1.0, 1.0, size=[TESTING_INPUT_SIZE, NOISE_SIZE])
	
	generated_patterns_train = generator.predict(random_inputs_train)
	generated_patterns_test = generator.predict(random_inputs_test)
	
	input_train, labels_train = chord_data_set(TRAINING_INPUT_SIZE, all_major=True)
	input_test, labels_test = chord_data_set(TESTING_INPUT_SIZE, all_major=True)
	
	all_inputs = np.concatenate((generated_patterns_train, input_train))
	all_labels = np.concatenate((np.ones([TRAINING_INPUT_SIZE, 1]), np.zeros([TRAINING_INPUT_SIZE, 1])))
	
	all_inputs_test = np.concatenate((generated_patterns_test, input_test))
	all_labels_test = np.concatenate((np.ones([TESTING_INPUT_SIZE, 1]), np.zeros([TESTING_INPUT_SIZE, 1])))

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
	desired_outputs_test = np.zeros([TESTING_INPUT_SIZE, 1])
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