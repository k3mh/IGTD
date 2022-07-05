
from sklearn.datasets import make_gaussian_quantiles, make_hastie_10_2, make_classification, make_moons
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import importlib
import numpy as np

importlib.reload(px)


"""Madelon rule example"""
from sklearn.datasets import  make_classification

temp_arr  = make_classification( n_samples=10000, n_features=2,  n_informative=2, n_redundant=0, n_repeated=0, class_sep=3, n_clusters_per_class=2)
temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                        "x2": temp_arr[0].transpose()[1], \
                        "y": temp_arr[1]})
temp_pd.y = temp_pd.y.astype(str)
fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
fig.update_layout(
    title={
        'text': "Simple example of 2 informative features with 2 clusters per class",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

""" Madelon rule end"""

""" make blobs example """
from sklearn.datasets import make_blobs
temp_arr  = make_blobs( n_samples=10000, n_features=2, centers=6)
temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                        "x2": temp_arr[0].transpose()[1], \
                        "y": temp_arr[1]})

# temp_pd.y = temp_pd.y.map( lambda x: 0 if x in (1,2,3) else 1)
temp_pd.y = temp_pd.y.astype(str)
fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
fig.update_layout(
    title={
        'text': "Simple example of 2 class problem based on 2 features",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

""" end of make blobs"""


""" make moons example """
from sklearn.datasets import make_blobs
temp_arr  = make_moons( n_samples=10000, random_state=100, noise = 0.1)
temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                        "x2": temp_arr[0].transpose()[1], \
                        "y": temp_arr[1]})

# temp_pd.y = temp_pd.y.map( lambda x: 0 if x in (1,2,3) else 1)
temp_pd.y = temp_pd.y.astype(str)
fig = px.scatter(data_frame = temp_pd , x="x1",  y="x2", color="y")
fig.update_layout(
    title={
        # 'text': "Simple example of 2 class problem based on 2 features",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

""" end of make blobs"""

""""dataset accuracy plot : start"""
fig = px.line(x=range(1, len(accuracy_lst)+1), y=accuracy_lst, markers=True, range_x= range(1, len(accuracy_lst)+2))
fig.update_layout(
    title={
        # 'text': "Datasets performance based on individual rules.",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
""""dataset accuracy plot : end """


""" Dataset combinations and selection"""
minimum_threshold = 0.95
dataset_comb_results = pd.read_csv("4082_dataset_comb_results_21May2022.csv")
dataset_comb_results = dataset_comb_results.loc[dataset_comb_results.auc >= minimum_threshold]
dataset_comb_results.sort_values("auc", inplace=True, ascending=False)
dataset_comb_results.drop(columns=["Unnamed: 0"], inplace=True)
dataset_comb_results = dataset_comb_results.reset_index(drop=True).reset_index()
selected_inds=[]
for i in [0, 10, 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100]:
    selected_inds.append(dataset_comb_results.iloc[(dataset_comb_results.accuracy - np.percentile(
        dataset_comb_results.accuracy, i)).abs().argsort()[:1]].index.values[0])
dataset_comb_results = dataset_comb_results.rename(columns={"index":"Dataset Index", "accuracy":"Accuracy", "auc":"AUC"})
fig = px.scatter(data_frame=dataset_comb_results, x="Dataset Index", y="AUC" )
fig.data[0].update(selectedpoints=selected_inds, selected=dict(marker=dict(color='purple', size=11)))

fig.show()


####################### OPS surface function #############
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def z_func(cov, com):
    z=np.sqrt(np.power(100-cov,2) + np.power(com, 2)) -1
    return z

OPS=[]
coverage = []
complexity = []
cov = np.linspace(0, 100, 30)
comp = np.linspace(1, 100, 30)

coverage, complexity = np.meshgrid(cov, comp)
OPS = z_func(coverage, complexity)


# Read data from a csv
# z_data = pd.DataFrame({"OPS": OPS, "Complexity":complexity, "Coverage":coverage}, index=range(len(OPS)))

fig = go.Figure(data=[go.Surface(x=coverage, y=complexity, z=OPS)])

fig.update_layout(title='OPS Function Surface', autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene=dict(
                      xaxis_title='Coverage',
                      yaxis_title='Complexity',
                      zaxis_title='OPS',
                  ),
                  )

fig.show()








