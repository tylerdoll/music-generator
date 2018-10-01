# 3rdparty
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.utils import np_utils
from music21 import converter, instrument, note, chord
import numpy as np

# standard
import glob
import pickle


def get_notes():
    notes = []
    notes_file = 'notes.pkl'

    try:
        with open(notes_file, 'rb') as f:
            notes = pickle.load(f)
    except FileNotFoundError:
        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            instruments = instrument.partitionByInstrument(midi)

            if instruments:
                notes_to_parse = instruments.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        with open(notes_file, 'wb') as f:
            pickle.dump(notes, f)

    return notes


def prepare_sequences(notes, n_vocab, sequence_length):
    """ Prepare data to train on.

    Args:
        sequence_length (int): number of notes needed to predict next note

    Returns:
        normalized input data for model training
    """
    pitches = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[c] for c in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize
    normalized_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)
    return normalized_input, network_output


def create_model(training_data, n_vocab,
                 dropout_rate=0.3, activation_func='softmax'):
    """ Create a model to train on.

    TODO: Play around with the layers to see if I can get better results

    Args:
        training_data: data to train on
        n_vocab (int): number of output nodes
        dropout_rate (float): rate to apply to dropout layers
        activation_func (str): function to use in activation layer

    Returns:
        model to train on
    """
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(training_data.shape[1], training_data.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_vocab))
    model.add(Activation(activation_func))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def train(model, x, y):
    filepath = 'weights/weights-imrpovement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks = [checkpoint]
    model.fit(x, y, epochs=200, batch_size=64, callbacks=callbacks)


if __name__ == "__main__":
    print('Getting notes...')
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f'Got {n_vocab} notes')

    print('Preparing sequences...')
    training_data, output_data = prepare_sequences(
        notes, n_vocab, sequence_length=100)

    print('Creating model...')
    model = create_model(training_data, n_vocab)

    print('Training model...')
    train(model, training_data, output_data)

    print('Done')
