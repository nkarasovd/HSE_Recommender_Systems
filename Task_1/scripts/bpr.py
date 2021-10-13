import numpy as np
import pandas as pd

from scripts.matrix_factorization import PairwiseRankModel


class BPRModel(PairwiseRankModel):
    def __init__(self, data_frame: pd.DataFrame, latent_size: int = 64,
                 learning_rate: float = 1e-3, lambda_: float = 1e-5,
                 epsilon: float = 1e-8, max_iterations: int = int(1e4),
                 verbose_step: int = int(1e3), sample_size: int = 5):
        super().__init__(data_frame, latent_size, learning_rate, lambda_, epsilon,
                         max_iterations, verbose_step, sample_size)

    def fit(self) -> 'BPRModel':
        self._reset_train_log()

        for i in range(self.max_iterations):
            for user in self.unique_users:
                pos_items, neg_items = self.users_interactions[user]

                for pos_item in pos_items:
                    neg_sample = np.random.choice(neg_items, size=self.sample_size, replace=False)
                    r_u_pos = self.W[user, :] @ self.H[:, pos_item]
                    for neg_item in neg_sample:
                        r_u_neg = self.W[user, :] @ self.H[:, neg_item]

                        loss = 1.0 / (1.0 + np.exp(r_u_pos - r_u_neg))

                        self._update_params(user, pos_item, neg_item, loss)

            if i == 0 or (i + 1) % self.verbose_step == 0:
                self._update_train_log(i + 1)
                if self._convergence_test(self.train_log[i + 1]):
                    break

        self._calculate_target_matrix()

        return self
