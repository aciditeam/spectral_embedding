#!/usr/bin/env python
# -*- coding: utf8 -*-

# Build the database for learning spectral embedding :
#   - generate random symbolic vectors
#   - generate corresponding wave files
#    

import cPickle as pickle
import random
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
from acidano.visualization.waveform.visualize_waveform import visualize_waveform

############################################################
############################################################
############################################################
# BUILD_DATABASE
def build_database(SYMBOLIC_METADATA_PATH, SOUND_DB_PATH, N_SAMPLES, MEAN_NUMBER_NOTE_PLAYED, INTENSITIES):
    ############################################################
    # Auxiliary functions
    def build_symbolic_database(N_orchestra):
        # Number of notes played at each time is modeled by poisson distrib
        number_instru_on = np.random.poisson(lam=MEAN_NUMBER_NOTE_PLAYED,size=N_SAMPLES)
        source_symbolic = np.zeros((N_SAMPLES, N_orchestra))
        intensity_grid = INTENSITIES.keys()
        intensity_number = len(intensity_grid)
        for n in range(N_SAMPLES):
            pitch_on = set()
            while len(pitch_on) < number_instru_on[n]:
                pitch_on.add(np.random.randint(N_orchestra))
            for p in pitch_on:
                source_symbolic[n,p] = intensity_grid[random.randrange(intensity_number)]
        return source_symbolic

    def find_wave(instrument_name, pitch, intensity):
        # Find notation
        mapping = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
        note_str = mapping[(pitch % 12)]
        note_str += str(pitch // 12)  # Octave
        intensity_str = INTENSITIES[intensity]
        # Grouping, since wave are split
        wave_path = SOUND_DB_PATH + '/' + instrument_name + '/' + note_str + '_' + intensity_str + '.wav'
        ###########
        ###########
        ###########
        ###########
        indisof = random.randint(0,5)
        if indisof == 0:
            wave_path = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_0.9c_HQ/Strings/Viola/ordinario/Va-ord-G#6-pp-1c.wav'
        elif indisof == 1:
            wave_path = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_0.9c_HQ/Strings/Violin/ordinario/Vn-ord-B4-ff-2c.wav'
        elif indisof == 2:
            wave_path = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_0.9c_HQ/Winds/Bassoon/ordinario/Bn-ord-C4-ff.wav'
        elif indisof == 3:
            wave_path = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_0.9c_HQ/Brass/Horn/ordinario/Hn-ord-G3-pp.wav'
        elif indisof == 4:
            wave_path = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_0.9c_HQ/Winds/Oboe/ordinario/Ob-ord-C5-mf.wav'
        elif indisof == 5:
            wave_path = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_0.9c_HQ/Brass/Trumpet-C/ordinario/TpC-ord-E4-pp.wav'
        ###########
        ###########
        ###########
        ###########
        # Normalize intensity
        return wave_path

    ############################################################
    # Load the symbolic orchestral mapping used for LOP (so extracted from the database, where unseen pitches have been removed)
    metadata = pickle.load(open(SYMBOLIC_METADATA_PATH,'rb'))
    instru_mapping = metadata['instru_mapping']
    N_orchestra = metadata['N_orchestra']

    # Build N random symbolic vectors
    symbolic = build_symbolic_database(N_orchestra)

    # For each
    for n in range(N_SAMPLES):
        waveforms = []
        # Find pitches and instruments associated to the symbolic vector and write the wav in a numpy array
        for instrument_name, ranges in instru_mapping.iteritems():
            index_min = ranges['index_min']
            index_max = ranges['index_max']
            pitch_min = ranges['pitch_min']
            for index in range(index_min, index_max):
                # Intensity > 0 => note played
                intensity = symbolic[n,index]
                if intensity > 0:
                    # Find its pitch
                    pitch = pitch_min + index - index_min
                    # Find corresponding wave
                    wave_path = find_wave(instrument_name, pitch, intensity)
                    # Read wave
                    (rate,waveform) = wav.read(wave_path)
                    waveforms.append(waveform)

        # Sum the waveforms together
        max_length = max([len(w) for w in waveforms])
        waveform_sum = np.zeros(max_length)
        N_waveform = len(waveforms)
        for w in waveforms:
            waveform_sum[:len(w)] += w//N_waveform
        # Compute MFCC
        # mfcc_coeff = mfcc(waveform_sum,samplerate=rate)
        mfcc_coeff = mfcc(waveform_sum,samplerate=rate,winlen=0.025,winstep=0.01,numcep=20,
                          nfilt=52,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,
                          ceplifter=22,appendEnergy=True)

        # Fit GMM

        # Debug plot
        # Waveform
        visualize_waveform(waveform_sum, '../DEBUG', 'waveform', subsampling_before_plot=1)
        # MFCCs
        visualize_mat(mfcc_coeff, '../DEBUG/', 'mfcc_coeff')

    return symbolic, mfcc_coeff


if __name__ == '__main__':
    import glob
    import re
    import shutil
    top_folder = '/home/aciditeam-leo/Aciditeam/database/Spectral_embedding/SOL_processed'
    wav_paths = glob.glob(top_folder + '/**/**/*.wav')

    skip = 20
    counter = 0
    for path_name in wav_paths:
        if counter % skip == 0:
            # Get filename
            splitted = (re.split(r'/', path_name))
            fname = (re.split(r'\.', splitted[-1]))[0]
            # Remove sharps (raise URL problem)
            fname = re.sub('#', 'D', fname)
            instru_name = splitted[-2]
            # Get wave and mfcc
            (rate,waveform) = wav.read(path_name)
            # mfcc_coeff = mfcc(waveform,samplerate=rate)
            mfcc_coeff = mfcc(waveform,samplerate=rate,winlen=0.025,winstep=0.01,numcep=52,
                              nfilt=52,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,
                              ceplifter=22,appendEnergy=True)
            # plot
            visualize_mat(mfcc_coeff, '../DEBUG/', instru_name + '_' + fname)
            # Copy wav
            shutil.copy(path_name, '../DEBUG/' + instru_name + '_' + fname + '.wav')
        counter += 1
