
import random
from random import randint
from numpy import array
NOTE_SPACE = 24		#two octaves of notes are valid here.

def get_three_notes_and_is_chord():
	answer = [0] * NOTE_SPACE
	if(randint(0, 1) == 0):		#make half major, half garbage
		#then make numbers in a major chord
		base = randint(0, 16)
		answer[base] = 1
		answer[base + 4] = 1
		answer[base + 7] = 1
		is_chord = 0
	else:
		#then just get three random numbers
		nums = random.sample(range(0, 24), 3)
		for n in nums:
			answer[n] = 1
		is_chord = 1
	return array(answer), array(is_chord)
	
	
def chord_data_set(size):
	data = []
	labels = []
	for i in range(size):
		chord, output = get_three_notes_and_is_chord()
		data.append(chord)
		labels.append(output)

	return array(data), array(labels)
		