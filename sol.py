# Predicting the Efficiency of Solar Cells From Molecular Structure
# Model Runner
#
# SMILES Parsing using Keras-Molecules https://github.com/maxhodak/keras-molecules
#
# Made by Liam Nakagawa 2017
gen = False

from keras.callbacks import ModelCheckpoint, TensorBoard
from molecules.vectorizer import SmilesDataGenerator
import pandas as pd
import os
if gen:
   from solrnn import SolModel
else:
   from solmodel import SolModel

num_smiles = 2322849
num_epochs = 50 if not gen else 5 #
epoch_size = 25000 if not gen else 1250 #Use 2320000 to run on entire CEPDB
batch_size = 2500 if not gen else 125#used to be 5000
latent_dim = 292
max_len = 120
test_split = 0.20
model_name = "convmode22.h5"
data_name = "cepdb.h5" #CEPDB SMILES + PCE


data = pd.read_hdf(data_name, 'table')
smiles = data['smiles']
pce = data['pce']

#Using Keras-Molecules SMILES Processing
datobj = SmilesDataGenerator(smiles, pce, max_len, test_split=test_split)
test_divisor = int((1 - datobj.test_split) / (datobj.test_split))
train_gen = datobj.train_generator(batch_size)
test_gen = datobj.test_generator(batch_size)
#Reset generators to not produce weights
if gen:
   train_gen = ((pces, smls) for (smls, pces, weights) in train_gen)
   test_gen = ((pces, smls) for (smls, pces, weights) in test_gen)
else:
   train_gen = ((smls, pces) for (smls, pces, weights) in train_gen)
   test_gen = ((smls, pces) for (smls, pces, weights) in test_gen)

#Run Model
model = SolModel()
if os.path.isfile(model_name):
   model.load(len(datobj.chars), model_name)
else:
   model.create(len(datobj.chars))

checkpointer = ModelCheckpoint(filepath = model_name,
                              verbose = 1,
                              save_best_only = True)

tboard = TensorBoard(log_dir='./logs',
                        histogram_freq=0,
                        write_graph=True,
                        write_images=False) #DataViz

model.sol.fit_generator(
   train_gen,
   (1-test_split)*epoch_size,
   nb_epoch = num_epochs,
   callbacks = [checkpointer,tboard],
   validation_data = test_gen,
   nb_val_samples = epoch_size * test_split,
   pickle_safe = True
)
