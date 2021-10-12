from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn")


class RecommendationModel(ABC):
    @abstractmethod
    def similar_items(self, item: int, count: int):
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user: int, count: int):
        raise NotImplementedError


class BaseModel(RecommendationModel, ABC):
    def __init__(self, data_frame: pd.DataFrame, latent_size: int,
                 learning_rate: float, lambda_: float, epsilon: float,
                 max_iterations: int, verbose_step: int):
        self.data_frame = data_frame
        self.numpy_data = data_frame.to_numpy()

        self.n_examples = self.numpy_data.shape[0]

        self.user_ids = self.numpy_data[:, 0] - 1
        self.movie_ids = self.numpy_data[:, 1] - 1
        self.ratings = self.numpy_data[:, 2]
        # nunique() != max(), some id may not be present
        self.total_users = data_frame.user_id.max()
        self.total_items = data_frame.movie_id.max()
        # save unique ids for users and items
        self.unique_users = data_frame.user_id.unique() - 1
        self.unique_items = data_frame.movie_id.unique() - 1

        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.lambda_ = lambda_  # regularization
        self.epsilon = epsilon  # convergence parameter
        self.max_iterations = max_iterations
        self.verbose_step = verbose_step

        self.support_right_board = 1.0 / np.sqrt(self.latent_size)
        # uniform distribution with support = [0, 1 / \sqrt{latent_size}]
        self.W = np.random.random(size=(self.total_users, self.latent_size)) * self.support_right_board
        self.H = np.random.random(size=(self.latent_size, self.total_items)) * self.support_right_board

        # target matrix
        self.V = None

        # save iterations and rmse values
        self.train_log = {}

    def _calculate_rmse(self) -> float:
        r_hat = self.V[self.user_ids, self.movie_ids]
        return np.linalg.norm(r_hat - self.ratings) / self.n_examples

    def _convergence_test(self, rmse: float) -> bool:
        return rmse < self.epsilon

    def _update_train_log(self, iter_num: int):
        self._calculate_target_matrix()
        rmse = self._calculate_rmse()
        print(f"> {iter_num:<10} | RMSE - {round(rmse, 7)}")
        self.train_log[iter_num] = rmse

    def _reset_train_log(self):
        self.train_log = {}

    def similar_items(self, item: int, count: int = 10) -> List[Tuple[int, float]]:
        ratings = [(other_item, np.linalg.norm(self.H[:, item - 1] - self.H[:, other_item - 1]))
                   for other_item in self.unique_items]
        items_ratings = sorted(ratings, key=lambda x: x[1])
        return items_ratings[:count]

    def recommend(self, user: int, count: int = 10) -> List[Tuple[int, int]]:
        user_items = self.data_frame.loc[self.data_frame["user_id"] == user]["movie_id"]
        unused_items = np.array(list(set(self.unique_items) - set(user_items)))
        ratings = self.V[user - 1][unused_items - 1]
        return sorted(list(zip(unused_items, ratings)), key=lambda x: -x[1])[:count]

    def plot_train_log(self):
        if not self.train_log:
            raise ValueError("Empty train log!")

        x = list(self.train_log.keys())
        y = list(self.train_log.values())

        plt.plot(x, y)

        plt.xlabel("Iteration number")
        plt.ylabel("RMSE value")

        plt.tight_layout()
        plt.show()

    @abstractmethod
    def _calculate_target_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self) -> 'BaseModel':
        raise NotImplementedError


class PairwiseRankModel(BaseModel, ABC):
    def __init__(self, data_frame: pd.DataFrame, latent_size: int,
                 learning_rate: float, lambda_: float,
                 epsilon: float, max_iterations: int,
                 verbose_step: int, sample_size: int):
        super().__init__(data_frame, latent_size, learning_rate, lambda_,
                         epsilon, max_iterations, verbose_step)
        self.sample_size = sample_size
        self.users_interactions = self._get_users_interactions()

    def _calculate_target_matrix(self):
        self.V = self.W @ self.H

    def _get_users_interactions(self) -> Dict[int, Tuple[List[int], List[int]]]:
        users_interactions = {}

        for user in self.unique_users:
            pos_items = self.numpy_data[self.user_ids == user][:, 1]
            neg_items = np.array(list(set(self.unique_items) - set(pos_items))) - 1
            users_interactions[user] = (pos_items - 1, neg_items)

        return users_interactions

    def _update_params(self, user: int, pos_item: int, neg_item: int, loss: float):
        w_u, y_pos, y_neg = deepcopy(self.W[user, :]), self.H[:, pos_item], self.H[:, neg_item]

        self.W[user, :] -= self.learning_rate * (loss * (y_neg - y_pos) + self.lambda_ * w_u)
        self.H[:, pos_item] -= self.learning_rate * (self.lambda_ * y_pos - loss * w_u)
        self.H[:, neg_item] -= self.learning_rate * (loss * w_u + self.lambda_ * y_neg)

    @abstractmethod
    def fit(self) -> 'PairwiseRankModel':
        raise NotImplementedError
