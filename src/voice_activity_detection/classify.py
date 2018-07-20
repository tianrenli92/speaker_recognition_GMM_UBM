from os.path import dirname,abspath,join
import pickle
import pandas as pd
from sklearn import tree

DATA = join(dirname(dirname(abspath(__file__))), 'data', 'testwav','0713','features_df_1s.pickle')
MODEL = join(dirname(dirname(abspath(__file__))), 'data', 'noise-train','model_tree_1s_bal.pickle')

with open(DATA, "rb") as file:
    test_df=pickle.load(file)

with open(MODEL, "rb") as file:
    classifier=pickle.load(file)

test_df = test_df[pd.notnull(test_df['RSE'])]
test_df = test_df[pd.notnull(test_df['RSE_std'])]
test_df['is_silent'] = test_df['RMS'] < 103.6
test_df_features=test_df.drop(['RMS', 'SE', 'type', 'name', 'number', 'is_silent'], axis=1)
test_predictions = classifier.predict(test_df_features)
test_df['is_speech']=test_predictions==2
print(test_df[test_df['is_silent']==False][['name','number','is_speech']])
