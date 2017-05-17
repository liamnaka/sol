#EXPERIMENTAL - Doesn't work that well currently
#Made by Liam Nakagawa 2017

from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from solmodel import SolModel
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from molecules.vectorizer import SmilesDataGenerator

data_name = 'cepdb.h5'
model_name = 'solmodel.h5'

model = SolModel()

def read_smiles_data(filename):
    import pandas as pd
    h5f = pd.read_hdf(filename, 'table')
    data = h5f['smiles'][:]
    # import gzip
    # data = [line.split()[0].strip() for line in gzip.open(filename) if line]
    return data

structures = read_smiles_data(data_name)

datobj = SmilesDataGenerator(structures, structures, 120)
train_gen = datobj.generator(20)


if os.path.isfile(model_name):
   model.load(len(datobj.chars), model_name)
else:
   raise ValueError("Model file %s doesn't exist" % model_name)

true_pred_gen = (((mat, weight, model.sol.predict(mat))
                     for (mat, _, weight) in train_gen))
text_gen = ((str.join('\n',
                     [str((datobj.table.decode(true_mat[vec_ix])[:np.argmin(weight[vec_ix])],
                           datobj.table.decode(vec)[:]))
                     for (vec_ix, vec) in enumerate(pred_mat)]))
            for (true_mat, weight, pred_mat) in true_pred_gen)
for _ in range(20):
   print(text_gen.next())

#print(model.sol.predict(train_gen))
#
#true_pred_gen = (((mat, weight, model.sol.predict(mat))
 #                     for (mat, _, weight) in train_gen))
    #for _ in range(args.sample):
#print(str(model.sol.predict(test_gen))+str(test_gen))
