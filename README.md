# Predicting the efficiency of solar cells with Deep Convnets
The deep neural model specified here learns the relationship between solar cell structure (**SMILES**) and efficiency (**PCE**) using one-dimensional convolutional layers.

Improving the efficiency of organic photovoltaics (**OPV**s) is an important research objective in the quest to produce cheap, cost-effective solar cells.

The simple, deep architecture specified here can learn to accurately predict the efficiency of OPVs when provided only the candidate polymers' molecular structures (and PCE labels for training).

The relative success of this model when trained on a dataset of over [2.3 M *molecular structure-power conversion efficiency* pairs](http://cleanenergy.molecularspace.org) suggests that neural networks are a potentially powerful tool for the discovery of highly efficient OPVs — a promsing source of renewable energy.



The parsing of SMILES strings is done using [Keras-Molecules](https://github.com/maxhodak/keras-molecules), go check it out!

