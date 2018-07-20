from sklearn.tree import _tree
import json
from os.path import join, dirname, abspath
import pickle
import numpy as np

FOLDER = join(dirname(dirname(abspath(__file__))), 'data', 'noise-train')
MODEL = join(FOLDER, 'model_tree_1s.pickle')
JSON_FILE = join(FOLDER, 'model_tree_1s.json')
feature_names = "ZCR,LEFR,SF,SF_std,SRF,SRF_std,SC,SC_std,BW,BW_std,NWPD,NWPD_std,RSE,RSE_std".split(",")

with open(MODEL, "rb") as file:
    classifier = pickle.load(file)

tree_ = classifier.tree_

feature_name = [
    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    for i in tree_.feature
]
print("def tree({}):".format(", ".join(feature_names)))

json_file = {}


def recurse(node, depth, json_file):
    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        json_file["feature"] = name
        json_file["threshold"] = threshold
        #        json_file["decision"] = None
        print("{}if {} <= {}:".format(indent, name, threshold))
        try:
            temp = json_file["left"]
        except:
            json_file["left"] = {}
        recurse(tree_.children_left[node], depth + 1, json_file["left"])
        print("{}else:  # if {} > {}".format(indent, name, threshold))
        try:
            temp = json_file["right"]
        except:
            json_file["right"] = {}
        recurse(tree_.children_right[node], depth + 1, json_file["right"])
    else:
        print("{}return {}".format(indent, tree_.value[node]))
        json_file["decision"] = bool(np.argmax(tree_.value[node]) == 1)
        json_file["false_samples"] = list(tree_.value[node][0])[0]
        json_file["true_samples"] = list(tree_.value[node][0])[1]
        #        json_file["threshold"] = 0.0
        #        json_file["feature"] = None
        #        json_file["left"] = None
        #        json_file["right"] = None
        return json_file


recurse(0, 1, json_file)
print(json.dumps(json_file, indent=4))
with open(JSON_FILE, "w") as file:
    json.dump(json_file, file, indent=4)
