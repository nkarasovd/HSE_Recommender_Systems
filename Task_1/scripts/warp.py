from typing import Tuple

import numpy as np
import pandas as pd

from scripts.matrix_factorization import PairwiseRankModel


class WARPModel(PairwiseRankModel):
    def __init__(self, data_frame: pd.DataFrame, latent_size: int = 64,
                 learning_rate: float = 1e-3, lambda_: float = 1e-5,
                 epsilon: float = 1e-8, max_iterations: int = int(1e4),
                 verbose_step: int = int(1e3), sample_size: int = 5):
        super().__init__(data_frame, latent_size, learning_rate, lambda_, epsilon,
                         max_iterations, verbose_step, sample_size)

    def _get_max_neg_item(self, user: int, pos_item: int) -> Tuple[int, float]:
        rating_pos = self.W[user, :] @ self.H[:, pos_item]

        user_neg_items = self.users_interactions[user][1]
        rank, sample_size = 0, min(len(user_neg_items), self.sample_size)

        neg_sample = np.random.choice(user_neg_items, size=sample_size, replace=False)
        ratings_neg = self.W[user, :] @ self.H[:, neg_sample]

        neg_item = neg_sample[-1]
        for rating_neg, neg_item in zip(ratings_neg, neg_sample):
            rank += 1
            if rating_pos < rating_neg + 1:
                break

        return neg_item, np.log(sample_size / rank)

    def fit(self) -> 'WARPModel':
        self._reset_train_log()

        for i in range(self.max_iterations):
            for user in self.unique_users:
                pos_items, neg_items = self.users_interactions[user]

                for pos_item in pos_items:
                    neg_item, loss = self._get_max_neg_item(user, pos_item)

                    self._update_params(user, pos_item, neg_item, loss)

            if i == 0 or (i + 1) % self.verbose_step == 0:
                self._update_train_log(i + 1)
                if self._convergence_test(self.train_log[i + 1]):
                    break

        self._calculate_target_matrix()

        return self
