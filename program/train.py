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
import numpy
from music21 import instrument, note, stream, chord, converter, duration

TIMESTEP = 0.25
SEQUENCE_LEN = int(8 / TIMESTEP)


def get_args():
    parser = argparse.ArgumentParser(description="Process model training info")
    parser.add_argument("--name", type=str, help="model name", required=True)
    parser.add_argument(
        "--songs_dir", type=str, help="midi songs for training", required=True
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to start training")
    parser.add_argument("--notes", type=str, default=None, help="preparsed notes")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for")
    return parser.parse_args()


def main():
    args = get_args()
    songs = glob.glob(args.songs_dir + "/*.mid")
    t = Trainer(args.name, songs, args.epochs)
    t.train_network(args.checkpoint, args.notes)


class Trainer:
    def __init__(self, model_name, songs, epochs):
        self.model_name = model_name
        self.songs = songs
        self.model = None
        self.epochs = epochs

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

        with open("notes/" + self.model_name, "wb") as filepath:
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
            epochs=self.epochs,
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


if __name__ == "__main__":
    main()
