#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for spectral embedding
#
# Warning !! Depend on the database used to train lop

import os
# Perso
from build_db import build_database

############################################################
############################################################
############################################################
# PARAMETERS FOR BUILD_DATABASE
CURRENT_DIR = os.getcwd()
N_SAMPLES = 10  # Number of samples used to generatethe embedded space
SOUND_DB_PATH = CURRENT_DIR + '/../../database/Spectral_embedding/SOL_0.9c_HQ'
SYMBOLIC_METADATA_PATH = CURRENT_DIR + '/../../lop/Data/metadata.pkl'
MEAN_NUMBER_NOTE_PLAYED = 5

# Intensity is discretized between ppp,    pp,    p,  mp,  mf, f, ff
#                                 0.125, 0.25,0.375, ....
INTENSITIES_MAPPING = {0.125: 'ppp', 0.25: 'pp', 0.375: 'p',
                       0.5: 'mp', 0.675: 'mf',
                       0.75: 'f', 0.875: 'ff', 1: 'fff',}


if __name__=='__main__':
    # Build the database : randomly generate symbolic vectors, compute associated GMM that represents MFCCs
    symbolic, gmm = build_database(SYMBOLIC_METADATA_PATH, SOUND_DB_PATH, N_SAMPLES, MEAN_NUMBER_NOTE_PLAYED, INTENSITIES_MAPPING)

    # Create pairs of similar/dissimilar based on nearest neighbours
    # Use siamese nets to build the embedded representation
