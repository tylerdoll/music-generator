
import random
from random import randint
from numpy import array
import numpy
NOTE_SPACE = 5		#5 octaves of notes are valid here.

RANDOMIZATION_MAX = 0.0

def get_chromatic_note_space(history_length, always_true=False):
	answer = numpy.random.uniform(low=0.0, high=RANDOMIZATION_MAX, size=(NOTE_SPACE * history_length,))

	if always_true or random.uniform(0, 1) < 0.5:
		
		base = randint(0, NOTE_SPACE)
		
		
		for i in range(history_length):
			start_of_this_frame = NOTE_SPACE * i
			
			offset = base + i
			#time_periods = get_chromatic_note_space[i]
			answer[start_of_this_frame + offset % NOTE_SPACE] = numpy.random.uniform(low=(1.0-RANDOMIZATION_MAX))

		label = 0
		
	else:
		for i in range(history_length):
			note = randint(0, NOTE_SPACE)
			answer[note] = numpy.random.uniform(low=(1.0-RANDOMIZATION_MAX))
		label = 1
		
	return array(answer), array(label)
	
	
def chromatic_data_set(size, history_length, always_true=False):
	data = []
	labels = []
	for i in range(size):
		#time_periods = [None] * history_length
		
		input, output = get_chromatic_note_space(history_length, always_true)

		data.append(input)
		labels.append(output)

	return array(data), array(labels)

print (chromatic_data_set(100, 2))