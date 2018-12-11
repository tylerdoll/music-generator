# -*- coding: utf-8 -*-
# std
import argparse
import glob
import pickle

# 3rd party
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Bidirectional
from keras.layers import GaussianNoise
import numpy
from music21 import instrument, note, stream, chord, converter, duration


def get_args():
    parser = argparse.ArgumentParser(description="Process model training info")
    parser.add_argument("mode", type=str, help="execution mode, train or predict")
    parser.add_argument("--name", type=str, help="model name")
    parser.add_argument("songs", type=str, help="midi songs for training")
    return parser.parse_args()


def main():
    args = get_args()


# Configuration
songs = glob.glob("midi_songs/*.mid")
TIMESTEP = 0.25
SEQUENCE_LEN = int(8 / TIMESTEP)
NUM_EPOCHS = 50
MODEL_NAME = f"custom-transposed-nulls-variableLength-ragtime-songs{len(songs)}-e{NUM_EPOCHS}-s{SEQUENCE_LEN}"


class Trainer:
    def __init__(self, model_name, songs):
        self.model_name = model_name
        self.songs = songs
        self.model = None

    def train_network(self, checkpoint=None, notes_file=None):
        """ Train a Neural Network to generate music """
        if not notes_file:
            notes = self.get_notes()
        else:
            with open(notes_file, "rb") as filepath:
                notes = pickle.load(filepath)

        # get amount of pitch names
        n_vocab = len(set(notes))
        print("n_vocab", n_vocab)

        network_input, network_output = self.prepare_sequences(notes, n_vocab)

        self.model = create_network(network_input, n_vocab)
        if checkpoint:
            print(f"Loading from checkpoint {checkpoint}")
            self.model.load_weights(checkpoint)

        self.train(network_input, network_output)
        file_name = self.model_name + ".hdf5"
        self.model.save(file_name)
        print(f"Model saved to {file_name}")

    def get_notes(self):
        """ Get all the notes and chords from the midi files in the ./midi_songs directory """
        notes = []

        for file in self.songs:
            print("Parsing %s" % file)
            try:
                midi = converter.parse(file)
            except IndexError as e:
                print(f"Could not parse {file}")
                print(e)
                continue

            notes_to_parse = None

            try:  # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            prev_offset = 0.0
            for element in notes_to_parse:
                if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                    duration = element.duration.quarterLength
                    if isinstance(element, note.Note):
                        name = element.pitch
                    elif isinstance(element, chord.Chord):
                        name = ".".join(str(n) for n in element.normalOrder)
                    notes.append(f"{name}${duration}")

                    rest_notes = int((element.offset - prev_offset) / TIMESTEP - 1)
                    for _ in range(0, rest_notes):
                        notes.append("NULL")

                prev_offset = element.offset

        print("notes", notes)

        with open("data/notes", "wb") as filepath:
            pickle.dump(notes, filepath)

        return notes

    def prepare_sequences(self, notes, n_vocab):
        """ Prepare the sequences used by the Neural Network """
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number + 1) for number, note in enumerate(pitchnames))
        note_to_int["NULL"] = 0

        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - SEQUENCE_LEN, 1):
            sequence_in = notes[i : i + SEQUENCE_LEN]
            sequence_out = notes[i + SEQUENCE_LEN]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, SEQUENCE_LEN, 1))
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = np_utils.to_categorical(network_output)

        return (network_input, network_output)

    def train(self, network_input, network_output):
        """ train the neural network """
        filepath = (
            self.model_name + "-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        )
        checkpoint = ModelCheckpoint(
            filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
        )
        callbacks_list = [checkpoint]
        self.model.fit(
            network_input,
            network_output,
            epochs=NUM_EPOCHS,
            batch_size=64,
            callbacks=callbacks_list,
        )


def create_network(network_input, n_vocab):
    print("Input shape ", network_input.shape)
    print("Output shape ", n_vocab)
    """ create the structure of the neural network """
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(512, return_sequences=True),
            input_shape=(network_input.shape[1], network_input.shape[2]),
        )
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model


t = Trainer(MODEL_NAME, songs)
t.train_network()


class Generator:
    def __init__(self, weights):
        self.weights = weights

    def generate(self):
        """ Generate a piano midi file """
        # load the notes used to train the model
        with open("data/notes", "rb") as filepath:
            notes = pickle.load(filepath)

        # Get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # Get all pitch names
        n_vocab = len(set(notes))
        network_input, normalized_input = self.prepare_sequences(
            notes, pitchnames, n_vocab
        )

        model = create_network(normalized_input, n_vocab)
        model.load_weights(self.weights)

        prediction_output = self.generate_notes(
            model, network_input, pitchnames, n_vocab
        )
        self.create_midi(prediction_output)

    def prepare_sequences(self, notes, pitchnames, n_vocab):
        """ Prepare the sequences used by the Neural Network """
        # map between notes and integers and back
        note_to_int = dict((note, number + 1) for number, note in enumerate(pitchnames))
        note_to_int["NULL"] = 0

        network_input = []
        output = []
        for i in range(0, len(notes) - SEQUENCE_LEN, 1):
            sequence_in = notes[i : i + SEQUENCE_LEN]
            sequence_out = notes[i + SEQUENCE_LEN]
            network_input.append([note_to_int[char] for char in sequence_in])
            output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        normalized_input = numpy.reshape(network_input, (n_patterns, SEQUENCE_LEN, 1))
        # normalize input
        normalized_input = normalized_input / float(n_vocab)

        return (network_input, normalized_input)

    def generate_notes(self, model, network_input, pitchnames, n_vocab):
        """ Generate notes from the neural network based on a sequence of notes """
        int_to_note = dict((number + 1, note) for number, note in enumerate(pitchnames))
        int_to_note[0] = "NULL"

        def get_start():
            # pick a random sequence from the input as a starting point for the prediction
            start = numpy.random.randint(0, len(network_input) - 1)
            pattern = network_input[start]
            prediction_output = []
            return pattern, prediction_output

        # generate verse 1
        verse1_pattern, verse1_prediction_output = get_start()
        for note_index in range(4 * SEQUENCE_LEN):
            prediction_input = numpy.reshape(
                verse1_pattern, (1, len(verse1_pattern), 1)
            )
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            index = numpy.argmax(prediction)
            print("index", index)
            result = int_to_note[index]
            verse1_prediction_output.append(result)

            verse1_pattern.append(index)
            verse1_pattern = verse1_pattern[1 : len(verse1_pattern)]

        # generate verse 2
        verse2_pattern = verse1_pattern
        verse2_prediction_output = []
        for note_index in range(4 * SEQUENCE_LEN):
            prediction_input = numpy.reshape(
                verse2_pattern, (1, len(verse2_pattern), 1)
            )
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            index = numpy.argmax(prediction)
            print("index", index)
            result = int_to_note[index]
            verse2_prediction_output.append(result)

            verse2_pattern.append(index)
            verse2_pattern = verse2_pattern[1 : len(verse2_pattern)]

        # generate chorus
        chorus_pattern, chorus_prediction_output = get_start()
        for note_index in range(4 * SEQUENCE_LEN):
            prediction_input = numpy.reshape(
                chorus_pattern, (1, len(chorus_pattern), 1)
            )
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            index = numpy.argmax(prediction)
            print("index", index)
            result = int_to_note[index]
            chorus_prediction_output.append(result)

            chorus_pattern.append(index)
            chorus_pattern = chorus_pattern[1 : len(chorus_pattern)]

        # generate bridge
        bridge_pattern, bridge_prediction_output = get_start()
        for note_index in range(4 * SEQUENCE_LEN):
            prediction_input = numpy.reshape(
                bridge_pattern, (1, len(bridge_pattern), 1)
            )
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            index = numpy.argmax(prediction)
            print("index", index)
            result = int_to_note[index]
            bridge_prediction_output.append(result)

            bridge_pattern.append(index)
            bridge_pattern = bridge_pattern[1 : len(bridge_pattern)]

        return (
            verse1_prediction_output
            + chorus_prediction_output
            + verse2_prediction_output
            + chorus_prediction_output
            + bridge_prediction_output
            + chorus_prediction_output
        )

    def create_midi(self, prediction_output):
        """ convert the output from the prediction to notes and create a midi file
            from the notes """
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            if "$" in pattern:
                pattern, dur = pattern.split("$")
                if "/" in dur:
                    a, b = dur.split("/")
                    dur = float(a) / float(b)
                else:
                    dur = float(dur)

            # pattern is a chord
            if ("." in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split(".")
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                new_chord.duration = duration.Duration(dur)
                output_notes.append(new_chord)
            # pattern is a rest
            elif pattern is "NULL":
                offset += TIMESTEP
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                new_note.duration = duration.Duration(dur)
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += TIMESTEP

        midi_stream = stream.Stream(output_notes)

        output_file = MODEL_NAME + ".mid"
        print("output to " + output_file)
        midi_stream.write("midi", fp=output_file)


g = Generator(MODEL_NAME + ".hdf5")
g.generate()
