import pandas as pd

from scripts.matrix_factorization import BaseModel
from copy import deepcopy


class ALSModel(BaseModel):
    def __init__(self, data_frame: pd.DataFrame, latent_size: int = 64,
                 learning_rate: float = 1e-3, lambda_: float = 1e-3,
                 epsilon: float = 1e-8, max_iterations: int = int(1e4),
                 verbose_step: int = int(1e3)):
        super().__init__(data_frame, latent_size, learning_rate, lambda_, epsilon,
                         max_iterations, verbose_step)

    def _calculate_target_matrix(self):
        self.V = self.W @ self.H

    def fit(self) -> 'ALSModel':
        self._reset_train_log()

        for i in range(self.max_iterations):
            self._calculate_target_matrix()

            error = deepcopy(self.V)
            error[self.user_ids, self.movie_ids] -= self.ratings

            if i % 2 == 0:
                self.W -= self.learning_rate * (error @ self.H.T + self.lambda_ * self.W)
            else:
                self.H -= self.learning_rate * (self.W.T @ error + self.lambda_ * self.H)

            if i == 0 or (i + 1) % self.verbose_step == 0:
                self._update_train_log(i + 1)
                if self._convergence_test(self.train_log[i + 1]):
                    break

        self._calculate_target_matrix()

        return self
