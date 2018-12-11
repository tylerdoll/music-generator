# music-generator
Machine Learning group project

Izaak Sulka, Tyler Doll and Jeff Greene

Included in this submission are two models, a fairly successful recurrent predictor and a
less successful GAN, along with the data you need to train them on.

Also includes a requirements.txt file for pip packages.


##############################################################################
				Bidrectional LSTM RNN
##############################################################################

Our RNN works pretty well at impersonating music.  There are two ways to interact with it.
You can train a model or use an existing one to predict.

Training:
Call `python3 train.py --name <name of model to train> --songs_dir <directory containing midis>`
Optional params include:
    `--checkpoint <hdf5 file>` to resume training from a previous checkpoint.
    `--notes <pickled notes file>` to use pre-parsed notes.
    `--epochs <int>` to set the number of epochs to train for, defaults to 10

Predicting/Generating:
Call `python3 predict.py <name of model>` where model name is:
    - ragtime
    - rap
    - christmas
Or a custom trained model.

Note that a hdf5 and notes file must exist with the same name to
generate a song.


##############################################################################
				Generative Adversarial Network
##############################################################################

Our GAN model was less successful than our other approach, but was a lot of work and
it still feels like there might be a way to get it to generate good impersonations
of music.  Included in this submission is a model in which the generator and discriminator
mostly balance out their learning, but don't quite make good music.  Everything needed
to run it and generate something is included here in the "GANs" subfolder.  You should
be able to just open a terminal next to the code (gan_final.py) and run 
"python gan_final.py"
and it will work.  However, you may want to edit the NUM_EPOCHS costant first to change
the amount of time it will take to train.  At the end of running, this code will produce
a .mid file which windows should know how to play from the default media player.  

Also included
