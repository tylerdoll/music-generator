
import random
from random import randint
from numpy import array
import numpy
NOTE_SPACE = 24		#two octaves of notes are valid here.
TRUE_CHORD_FALSE_MAX = 0.0

def get_three_notes_and_is_chord(all_major=False):
	answer = [0] * NOTE_SPACE
	if(randint(0, 1) == 0 or all_major):		#make half major, half garbage
	
		#return numpy.ones( NOTE_SPACE), array(0)
		
		#then make numbers in a major chord
		base = randint(0, 16)
		#base = 0
		#first we make random values less than the threshhold
		answer = numpy.random.uniform(low=0.0, high=TRUE_CHORD_FALSE_MAX, size=(NOTE_SPACE,))

		
		answer[base] = numpy.random.uniform(low=(1-TRUE_CHORD_FALSE_MAX))
		answer[base + 4] = numpy.random.uniform(low=(1-TRUE_CHORD_FALSE_MAX))
		answer[base + 7] = numpy.random.uniform(low=(1-TRUE_CHORD_FALSE_MAX))
		
		
		is_chord = 0
	else:
		raise ValueError("not supposed to be using this anymore")
		#then just get three random numbers
		
		# if it's not a chord, just do random noise
		answer = numpy.random.uniform( high=TRUE_CHORD_FALSE_MAX,  size=(NOTE_SPACE,))
		
		nums = random.sample(range(0, 24), 3)
		for n in nums:
			answer[n] = numpy.random.uniform(low=(1-TRUE_CHORD_FALSE_MAX))
		
		is_chord = 1
		
	return array(answer), array(is_chord)
	
	
def chord_data_set(size, all_major=False):
	data = []
	labels = []
	for i in range(size):
		chord, output = get_three_notes_and_is_chord(all_major)
		data.append(chord)
		labels.append(output)

	return array(data), array(labels)
		
        