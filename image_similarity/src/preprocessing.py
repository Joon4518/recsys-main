import os 
import sys
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import IncrementalPCA

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

class Similarity(IncrementalPCA):
    def __init__(self, n_components=None, batch_size=None):
        super().__init__()
        self.n_components = n_components
        self.batch_size=batch_size

    def fit_ipca(self, X):
        """
        Parameter
        ---------
        X : Array, shape (n_samples, n_features)
            Batch dataset consists of tensors.

        Return
        ------
        self
        """
        for i in range(len(X)):
            if len(X[i][0]) >= self.n_components:
                self.partial_fit(X[i][1])
    
    def make_batch_knn(self, X, K):
        """
        Parameters
        ----------
        X : Array, shape (n_samples, n_features)
            Batch dataset consists of tensors.
        
        K : int 
            A number of nearest neighbors.

        Returns
        -------
        batch_knn : Array, shape (n_batches, n_batches)
            Batch KNN object list.
        """

        batch_knn = []

        for i in range(len(X)):
            transformed_features = self.transform(X[i][1])
            user_ids = list(
                map(lambda x : x.decode("UTF-8"), X[i][0].numpy().tolist())
            )
            # 메모리에 다 안올라가는 데이터 셋 크기면 db나 pkl로 나눠 저장
            curr_user_index = {idx:id for idx, id in enumerate(user_ids)}

            nn = NearestNeighbors(
                n_neighbors= min(len(X[i][1]), K),
                radius=0.5,
                algorithm='brute',
                metric='cosine',
                )
            
            nn.fit(transformed_features)
            # Transform batch tensors of strings to list of strings
            
            batch_knn.append([nn, curr_user_index])
        
        return batch_knn

    def fit_knn(self, batch_knn, X, K):
        """
        Parameters
        ----------
        batch_knn : Array, shape (n_batches, n_batches)
            Batch KNN object list.

        X : Array, shape (n_samples, n_features)
            Batch dataset consists of tensors.

        K : int 
            Top K recommendation number.

        Returns
        -------

        """
        user_profile_similarity = {}

        for i in range(len(X)):
            user_ids = list(
                    map(lambda x : x.decode("UTF-8"), X[i][0].numpy().tolist())
                    )
            transformed_features = self.transform(X[i][1])

            for id, feature in zip(user_ids, transformed_features):
                batch_knn_res = []

                for knn, curr_user_index in batch_knn:
                    dist, idx = knn.kneighbors([feature])
                    dist, idx = dist[0], idx[0]

                    batch_knn_res.extend(
                        list(
                            zip(dist, map(lambda x: curr_user_index[x], idx))
                        )
                    )
                
                # DB에 바로 저장 고려
                user_profile_similarity[id] = sorted(batch_knn_res, key=lambda x : x[0])[:K*2]
        
        return user_profile_similarity