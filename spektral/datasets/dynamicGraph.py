import json
import os
import os.path as osp
import shutil

import numpy as np
import scipy.sparse as sp
from networkx.readwrite import json_graph
from itertools import islice
from sklearn.preprocessing import OneHotEncoder

from spektral.data import Dataset, GraphSnapshot, DynamicGraph 
from spektral.data.dataset import DATASET_FOLDER
from spektral.datasets.utils import download_file
from spektral.utils.logging import log


class MovieLen(Dataset):
    """
    **Arguments**

    """
    #url = "http://snap.stanford.edu/graphsage/{}.zip"
    url = "https://raw.githubusercontent.com/StatsDLMathsRecomSys/Self-attention-with-Functional-Time-Representation-Learning/master/input_data/ml-1m/"
    name = "MovieLen"
    files = ["movies.dat", "ratings.dat","users.dat"]

    def __init__(self, **kwargs):
        self.mask_tr = self.mask_va = self.mask_te = None
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, "Dynamic", self.name)


    def encode_movie(self, movies):
        gendre_dict = {"Action" : 0,
                       "Adventure":1,
                        "Animation" : 2,
                        "Children's" :3,
                        "Comedy":4,
                        "Crime":5,
                        "Documentary":6,
                        "Drama":7,
                        "Fantasy":8,
                        "Film-Noir":9,
                        "Horror":10,
                        "Musical":11,
                        "Mystery":12,
                        "Romance":13,
                        "Sci-Fi":14,
                        "Thriller":15,
                        "War":16,
                        "Western":17}
        movie_id_map = {}
        num_movies = movies.shape[0]

        encoded = np.zeros([num_movies,len(gendre_dict)+2])
        for i in range(num_movies):
            movie_id_map[movies[i,0]] = i
            encoded[i,0] = i
            encoded[i,1] = 1
            genre_list = movies[i,2].split("|")
            for g in genre_list:
                encoded[i, gendre_dict[g]+2] = 1

        return encoded, movie_id_map


    def encode_users(self, users, starting_id):
        # UserID::Gender::Age::Occupation::Zip-code
        num_users = users.shape[0]
        enc = OneHotEncoder().fit_transform(users[:,1:4]).toarray()
        user_list = np.zeros([num_users,1])
        users_id_map = {}
        for i in range(num_users):
            users_id_map[users[i, 0]] = starting_id
            user_list[i,0] = starting_id
            starting_id += 1
        encoded = np.concatenate((user_list, enc), axis=1)
        return encoded, users_id_map


    def read(self) -> DynamicGraph:
        '''
        '''
        movie_file = osp.join(self.path, self.files[0])
        rating_file = osp.join(self.path, self.files[1])
        user_file = osp.join(self.path, self.files[2])

        node_id = 0
        node_id_map = {}
        start_timestamp = None

        movies=None
        users=None

        edge_sequence = {}
        y_true_labels = {}
        
        print("\n\n**** Loading %s network ****" % (self.name))

        with open(movie_file, "r", encoding="ISO-8859-1") as f:
            movies = np.array([ tuple(l.strip().split("::")) for l in f.readlines()])
        # Format: MovieID::Title::Genres
        movie_features, movie_id_map = self.encode_movie(movies)

        with open(user_file, "r", encoding="ISO-8859-1") as f:
            users = np.array([ tuple(l.strip().split("::")) for l in f.readlines()])
        user_features , user_id_map = self.encode_users(users, movie_features.shape[0])

        c_movie_features = np.zeros((movie_features.shape[0], user_features.shape[1]-1))
        movies_f = np.hstack((movie_features[:, 1:], c_movie_features))
        c_user_features = np.zeros((user_features.shape[0], movie_features.shape[1]-1))
        users_f = np.hstack((c_user_features,  user_features[:, 1:]))
        nodes_feature = np.vstack((users_f, movies_f)) # nodes_feature[0] indicate whether it is a movie or user
        with open(rating_file, "r", encoding="ISO-8859-1") as f:
            f.readline()
            for cnt, l in enumerate(f):
                # FORMAT: UserID::MovieID::Rating::Timestamp
                ls = l.strip().split("::") 
                if start_timestamp is None:
                    start_timestamp = int(ls[3])
                t = int(ls[3]) - start_timestamp
                if t not in edge_sequence:
                    edge_sequence[t] = []
                current_user =user_id_map[ls[0]]               
                current_item = movie_id_map[ls[1]]
                edge_sequence[t].append((current_user, current_item, ls[2]))

        return [DynamicGraph(nodes_feature, edge_sequence)]
        

    def download(self):
        print("Downloading {} dataset.".format(self.name))
        for f in self.files:
            url = self.url+f
            download_file(url, self.path, f)
 

class METR_LA(Dataset):

    url="https://github.com/lehaifeng/T-GCN/archive/8427128f04157e6fd0b239a8734a468d923cd0c9.zip"
