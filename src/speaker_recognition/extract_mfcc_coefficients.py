import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import csv
import numpy as np
import pickle
from os.path import dirname, abspath, join
import pandas as pd

def save_mfcc(mfcc_coefficients, file_name):
    print("saving the mfcc coefficients...")
    # os.chdir(ROOT_SPEAKER_RECOGNITION)
    with open(file_name, "wb") as f:
        pickle.dump(mfcc_coefficients,f)


if __name__ == "__main__":
    DATA_FOLDER = join(dirname(dirname(abspath(__file__))), 'data', 'speaker-train', 'dev-clean')
    IS_SEPERATE = False
    flag = 0
    i = 0
    speaker_set=set()
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_name_list = file.split(".")
            if file_name_list[-1] == "wav":
                speaker_id=file.split("-")[0]
                if speaker_id in speaker_set:
                    break
                speaker_set.add(speaker_id)
                i += 1
                print(i, ":", file)
                (rate, sig) = wav.read(join(root, file))
                mfcc_features = mfcc(sig, rate)
                # print(type(mfcc_feat))
                if IS_SEPERATE:
                    save_mfcc(mfcc_features, join(root, file_name_list[0]))
                else:
                    if flag == 0:
                        mfcc_features_all = mfcc_features
                        flag = 1
                    else:
                        mfcc_features_all = np.append(mfcc_features_all, mfcc_features, axis=0)
                break
    mfcc_features_df=pd.DataFrame.from_dict(mfcc_features_all)
    if not IS_SEPERATE:
        save_mfcc(mfcc_features_df, join(DATA_FOLDER,'mfcc.pickle'))

    print("*********extraction done************")
