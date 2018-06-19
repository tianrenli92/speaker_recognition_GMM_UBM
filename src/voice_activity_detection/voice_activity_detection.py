#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:12:18 2018

@author: siva
"""

# creating dataset for voice acitivity detection
from .extract_features import extract_features
import numpy as np
import os
from os.path import dirname,abspath,join
import scipy.io.wavfile as wav

# import seaborn as sns
# import matplotlib.pyplot as plt
# import copy
# import random
# from scipy.stats import mode


DATA_FOLDER = join(dirname(dirname(abspath(__file__))),'data','noise-train')
FRAME_LENGTH = 25  # ms
WINDOW_LENGTH = 5  # s


os.chdir(DATA_FOLDER)
dataset_dict = {"ZCR": [], "RMS": [], "spectral_flux": [], \
                "spectral_centroid": [], "spectral_rolloff": [], \
                "bandwidth": [], "audio": [], "nwpd": [], "rse": []}
for root, dirs, files in os.walk(DATA_FOLDER):
    for audio in files:
        if "noise" in audio or "music" in audio or "speech" in audio or "audio" in audio:
            print("****************************")
            print("reading:", audio)
            sampling_rate, sig = wav.read(join(root, audio))
            print("sampling rate:", sampling_rate, "signal length", len(sig))
            index = 0
            while index + (sampling_rate * WINDOW_LENGTH) < len(sig):
                sample = sig[index:(index + (sampling_rate * WINDOW_LENGTH))]
                ef = extract_features(sample, FRAME_LENGTH, sampling_rate)
                ZCR, RMS, sf, sr, sc, bd, nwpd, rse = ef.return_()
                dataset_dict["ZCR"].append(ZCR)
                dataset_dict["RMS"].append(RMS)
                dataset_dict["spectral_flux"].append(np.mean(sf))
                dataset_dict["spectral_centroid"].append(np.mean(sc))
                dataset_dict["spectral_rolloff"].append(np.mean(sr))
                dataset_dict["bandwidth"].append(np.mean(bd))
                dataset_dict["nwpd"].append(np.mean(nwpd))
                dataset_dict["rse"].append(np.mean(rse))
                dataset_dict["audio"].append(audio.split("-")[0])
                # if audio.split("-")[0] == "noise":
                #     dataset_dict["audio"].append("r")  # RED -> NOISE
                # elif audio.split("-")[0] == "music":
                #     dataset_dict["audio"].append("g")  # GREEN -> MUSIC
                # elif audio.split("-")[0] == "speech":
                #     dataset_dict["audio"].append("b")  # BLUE -> SPEECH
                index += sampling_rate * WINDOW_LENGTH
                # if index == 5*sampling_rate*WINDOW_LENGTH:
                #    break
values = dataset_dict.values()
print([len(e) for e in values])
print("finished")
np.save(join(DATA_FOLDER,"features.npy"), dataset_dict)


