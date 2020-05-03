from __future__ import print_function
import importlib
import DataSetGen
import Evaluation
importlib.reload(DataSetGen)
importlib.reload(Evaluation)

import pandas as pd
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np

import lime
import lime.lime_tabular

from anchor import utils
from anchor import anchor_tabular

import collections
from numba import njit, prange
import numba




#@numba.jit(parallel=True)
def get_exp_lime(data, explainer, feature_names,  model, num_feature=2, top_labels=1):

    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])
    explainer_lib = "lime"
    for count, i in enumerate(data.index.to_list()):
        print(count, i)
        exp = explainer.explain_instance(data.loc[i], model.predict_proba, num_features=num_feature, top_labels=top_labels)
        for exp_list in exp.as_map().items():
            explaination_df =  explaination_df.append(pd.DataFrame({"explainer_lib": [explainer_lib], "instance": [i], "features": [[feature_names[item[0]] for item in exp_list[1]]], "importance":  [[item[1] for item in exp_list[1]]] }))

    return explaination_df

def get_exp_lime_parallel(data, explainer, feature_names,  model, num_feature=2, top_labels=1):

    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])
    explainer_lib = "lime"


    for count, i in enumerate(data.index.to_list()):
        print(count, i)
        exp = explainer.explain_instance(data.loc[i], model.predict_proba, num_features=num_feature, top_labels=top_labels)
        for exp_list in exp.as_map().items():
            explaination_df =  explaination_df.append(pd.DataFrame({"explainer_lib": [explainer_lib], "instance": [i], "features": [[feature_names[item[0]] for item in exp_list[1]]], "importance":  [[item[1] for item in exp_list[1]]] }))

    return explaination_df



# Anchor explainer

def get_exp_anchor(data, explainer, precision=.95, model=None):

    explainer_lib = "anchor"
    explaination_df = pd.DataFrame(columns=["explainer_lib", "instance", "features", "importance"])

    for count, i in enumerate(data.index.to_list()):
        print(count, i)
        explanation = explainer.explain_instance(data.loc[i].values, model.predict, threshold=precision)
        vars = list(map((lambda x: x.split()[0]), explanation.names()))
        print(explanation.names(), vars)
        explaination_df = explaination_df.append(pd.DataFrame(
            {"explainer_lib": [explainer_lib], "instance": [i], "features": [vars],
             "importance": [list(range(1, len(vars) + 1)) ] }))
    return  explaination_df

