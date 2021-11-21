
from sklearn.datasets import make_gaussian_quantiles, make_hastie_10_2, make_classification
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


"""Madelon rule example"""
from sklearn.datasets import  make_classification

temp_arr  = make_classification( n_samples=1000, n_features=2,  n_informative=2, n_redundant=0, n_repeated=0, class_sep=3, n_clusters_per_class=2)
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
temp_arr  = make_blobs( n_samples=1000, n_features=2, centers=6)
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


""" make blobs example """
from sklearn.datasets import make
temp_arr  = make_blobs( n_samples=1000, n_features=2, centers=6)
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