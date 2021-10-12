import numpy as np
import pandas as pd

from scripts.matrix_factorization import BaseModel


class SVDModel(BaseModel):
    def __init__(self, data_frame: pd.DataFrame, latent_size: int = 64,
                 learning_rate: float = 1e-3, lambda_: float = 1e-3,
                 lambda_bias: float = 1e-4, epsilon: float = 1e-8,
                 max_iterations: int = int(1e5), verbose_step: int = int(1e4)):
        super().__init__(data_frame, latent_size, learning_rate, lambda_,
                         epsilon, max_iterations, verbose_step)
        self.lambda_bias = lambda_bias

        # initialization of bias vectors
        self.mu = np.mean(self.ratings.mean())
        self.W_bias = np.random.normal(2.5, 0.5, self.total_users).reshape(-1, 1)
        self.H_bias = np.random.normal(2.5, 0.5, self.total_items).reshape(1, -1)

    def _calculate_target_matrix(self):
        self.V = self.W @ self.H + self.W_bias + self.H_bias + self.mu

    def fit(self) -> 'SVDModel':
        self._reset_train_log()

        for i in range(self.max_iterations):
            # use indexing from slides
            # select random V_{ij} from V (not empty)
            user, movie, rating = self.numpy_data[np.random.randint(self.n_examples), :]
            user -= 1
            movie -= 1

            # calculate error = W_{i*}H_{*j} - V_{ij}
            error = self.W[user, :] @ self.H[:, movie] + self.W_bias[user, :] + \
                    self.H_bias[:, movie] + self.mu - rating

            # update W_{i*} and H_{*j}
            # W_{i*} = W_{i*} - learning_rate \cdot (error \cdot H_{*j}^T + \lambda \cdot W_{i*})
            # H_{*j} = H_{*j} - learning_rate \cdot (error \cdot W_{i*}^T + \lambda \cdot H_{*j})
            cur_w_user = self.W[user, :].copy()
            self.W[user, :] -= self.learning_rate * \
                               (error * self.H[:, movie].T + self.lambda_ * self.W[user, :])
            self.H[:, movie] -= self.learning_rate * \
                                (error * cur_w_user.T + self.lambda_ * self.H[:, movie])

            # update bias
            self.mu -= self.learning_rate * error
            self.W_bias[user, :] -= self.learning_rate * (error + self.lambda_bias * self.W_bias[user, :])
            self.H_bias[:, movie] -= self.learning_rate * (error + self.lambda_bias * self.H_bias[:, movie])

            if i == 0 or (i + 1) % self.verbose_step == 0:
                self._update_train_log(i + 1)
                if self._convergence_test(self.train_log[i + 1]):
                    break

        self._calculate_target_matrix()

        return self
