#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:12:18 2018

@author: siva
"""

# creating dataset for voice acitivity detection
from extract_features import extract_features
import numpy as np
import os
from os.path import dirname, abspath, join
import scipy.io.wavfile as wav
import pandas as pd
import pickle

DATA_FOLDER = join(dirname(dirname(abspath(__file__))), 'data', 'testwav','0713')
WINDOW_LENGTH = 1
FRAME_LENGTH = 25



feature_name = "RMS,SE,ZCR,LEFR,SF,SF_std,SRF,SRF_std,SC,SC_std,BW,BW_std,NWPD,NWPD_std,RSE,RSE_std,type,name,number".split(
    ",")
features_dict = {feature: [] for feature in feature_name}

for root, dirs, files in os.walk(DATA_FOLDER):
    for audio in files:
        if "noise" in audio or "music" in audio or "speech" in audio or "audio" in audio:
            print("****************************")
            print("reading:", audio)
            sampling_rate, sig = wav.read(join(root, audio))
            print("sampling rate:", sampling_rate, "signal length", len(sig))
            index = 0
            number = 0
            while index + (sampling_rate * WINDOW_LENGTH) < len(sig):
                sample = sig[index:(index + (sampling_rate * WINDOW_LENGTH))]
                ef = extract_features(sample, FRAME_LENGTH, sampling_rate)
                rms, se, zcr, lefr, sf, srf, sc, bd, nwpd, rse = ef.return_()
                features_dict["RMS"].append(rms)
                features_dict["SE"].append(se)
                features_dict["ZCR"].append(zcr)
                features_dict["LEFR"].append(lefr)
                features_dict["SF"].append(np.mean(sf))
                features_dict["SF_std"].append(np.std(sf))
                features_dict["SC"].append(np.mean(sc))
                features_dict["SC_std"].append(np.std(sc))
                features_dict["SRF"].append(np.mean(srf))
                features_dict["SRF_std"].append(np.std(srf))
                features_dict["BW"].append(np.mean(bd))
                features_dict["BW_std"].append(np.std(bd))
                features_dict["NWPD"].append(np.mean(nwpd))
                features_dict["NWPD_std"].append(np.std(nwpd))
                features_dict["RSE"].append(np.mean(rse))
                features_dict["RSE_std"].append(np.std(rse))
                features_dict["type"].append(audio.split("-")[0])
                features_dict["name"].append(audio)
                features_dict["number"].append(number)
                number += 1
                index += sampling_rate * WINDOW_LENGTH

features_df = pd.DataFrame.from_dict(features_dict)
features_df = features_df[feature_name]

with open(join(DATA_FOLDER, "features_df_1s.pickle"), "wb") as file:
    pickle.dump(features_df, file)
