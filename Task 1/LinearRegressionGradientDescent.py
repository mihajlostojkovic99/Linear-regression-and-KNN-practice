import numpy as np
import pandas as pd

class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        # Mapiramo koeficijente u niz oblika (n + 1) x 1
        self.coeff = np.array(args).reshape(-1, 1)

    # Racuna se koristeci Mean-square error.
    # m - broj uzoraka (redova u DataFrame-u)
    # y_predicted - m x 1 niz predvidjenih cena uzoraka
    # y_target - m x 1 niz stvarnih cena uzoraka
    # MS_error = (1 / 2 * m) * sum ((y_predicted - y_target) ^ 2)
    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    # Argument mora biti DataFrame
    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    # Jedan korak u algoritmu gradijentnog spusta.
    def gradient_descent_step(self, learning_rate):
        # learning_rate - korak ucenja; dimenzije ((n + 1) x 1);
        # korak ucenja je razlicit za razlicite koeficijente
        # m - broj uzoraka
        # n - broj razlicitih atributa (osobina)
        # features – dimenzije (m x (n + 1));
        # n + 1 je zbog koeficijenta c0
        # self.coeff – dimenzije ((n + 1) x 1)
        # predicted – dimenzije (m x (n + 1)) x ((n + 1) x 1) = (m x 1)
        predicted = self.features.dot(self.coeff)
        # koeficijeni se azuriraju po formuli:
        # coeff(i) = coeff(i) - learning_rate * gradient(i)
        # za i-ti koeficijent koji mnozi i-ti atribut
        # gledaju se samo vrednosti i-tog atributa za sve uzorke
        # gradient(i) = (1 / m) * sum(y_predicted - y_target) * features(i)

        # (predicted - self.target) - dimenzije (m x 1)
        # features - dimenzije (m x (n + 1));
        # transponovana matrica ima dimenzije ((n + 1) x m)
        # gradient - dimenzije ((n + 1) x m) x (m x 1) = (n + 1) x 1
        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):
        # Istorija Mean-square error-a kroz iteracije gradijentnog spusta.
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    # features mora biti DataFrame
    def fit(self, features, target):
        self.features = features.copy(deep=True)
        # Pocetna vrednost za koeficijente je 0.
        # self.coeff - dimenzije ((n + 1) x 1)
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
        # Unosi se kolona jedinica za koeficijent c0,
        # kao da je vrednost atributa uz c0 jednaka 1.
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        # self.features - dimenzije (m x (n + 1))
        self.features = self.features.to_numpy()
        # self.target - dimenzije (m x 1)
        self.target = target.to_numpy().reshape(-1, 1)

