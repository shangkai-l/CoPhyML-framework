"""
Created on Feb 7 2025
@author: Shangkai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic
from denseweight import DenseWeight
from scipy.optimize import dual_annealing, differential_evolution, minimize
import matplotlib.pyplot as plt



def compute_result(true, predict):
    true = np.array(true).reshape(-1, 1)
    predict = np.array(predict).reshape(-1, 1)
    rmse = np.average((true - predict) ** 2) ** (1 / 2)
    diff = np.abs(true - predict)
    mape = np.mean(diff / np.abs(true))
    return rmse, mape


#  ——————————————————————————————————————————————————————————————————————load dataset
knowledge_label = 0
if knowledge_label == 0:
    read_path = r"D:\shangkai\tj9_ori.xlsx"
    data = pd.read_excel(read_path)
    X = np.array(data_pi.iloc[:, :-3])
    y = np.array(data_pi.iloc[:, -3:])
else:
    read_path = r"D:\shangkai\tj9_pi.xlsx"
    data = pd.read_excel(read_path)
    X = np.array(data_pi.iloc[:, :9])
    y = np.array(data_pi.iloc[:, 9:])

y_use = y.copy()
Test_size_ = 0.7
x_train, x_test, y_train, y_test = train_test_split(X, y_use, test_size=Test_size_, random_state=0, shuffle=False)

target = 1  # for thrust as target
#  ——————————————————————————————————————————————————————————————————————load dataset


#  ——————————————————————————————————————————————————————————————————————prepare models
"""
include The proposed methods, SVR, Gaussian Process Regression, ANN, DNN, Random forest, Xgboost
"""


# Method based on the integration layer scheme 1
class CustomObj:
    def __init__(self, epsilon=1.0, alpha=0.01, bounds=(-10000, 10000), feature_num=1,
                 optimizer='differential_evolution', max_iter=1000000, seed=1, weight=None,
                 initial_guess=None, x0=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.bounds = [bounds] * feature_num
        self.optimizer = optimizer
        self.coef_ = None
        self.max_iter = max_iter
        self.seed = seed
        self.message = None
        self.weight = weight
        self.initial_guess = initial_guess
        self.x0 = x0

    def custom_loss(self, params, x, y):
        predictions = np.dot(x, params)
        residuals = y - predictions

        epsilon_mse_loss = np.sum(np.maximum(0, residuals ** 2 - (self.epsilon) ** 2))  # Quadratic ε-Insensitive Loss
        L1_norm = np.sum(np.abs(params))  # L1 sparsity regularization

        total_loss = epsilon_mse_loss + self.alpha * L1_norm  # objective function
        return total_loss

    def fit(self, x, y):
        if self.optimizer == 'differential_evolution':
            if self.initial_guess is not None:
                result = differential_evolution(self.custom_loss, self.bounds, args=(x, y),
                                                maxiter=self.max_iter, seed=self.seed, tol=1e-10, init=initial_guess)
            else:
                result = differential_evolution(self.custom_loss, self.bounds, args=(x, y),
                                                maxiter=self.max_iter, seed=self.seed, tol=1e-10)
            result = differential_evolution(self.custom_loss, self.bounds, args=(x, y), maxiter=self.max_iter, seed=self.seed, tol=1e-10)
        elif self.optimizer == 'dual_annealing':
            result = dual_annealing(self.custom_loss, self.bounds, args=(x, y), maxiter=self.max_iter)
        elif self.optimizer == 'minimize':
            initial_guess = [0, 0] if self.fit_intercept else [0]
            result = minimize(self.custom_loss, initial_guess, args=(x, y), bounds=self.bounds, maxiter=self.max_iter)
        else:
            raise ValueError("Unsupported optimizer. Use 'differential_evolution' or 'minimize'.")
        self.coef_ = result.x

    def predict(self, x):
        if self.coef_ is None:
            raise ValueError(
                "The model is not trained yet. Call 'fit' with appropriate arguments before using this estimator.")
        return np.dot(x, self.coef_)


e_obj = CustomObj(epsilon=6.8, alpha=0.055, feature_num=np.shape(x_train)[1], optimizer='differential_evolution')


# Method based on the integration layer scheme 2
class CustomObj_w:
    def __init__(self, epsilon=1.0, alpha=0.01, bounds=(-1000, 1000), feature_num=1,
                 optimizer='differential_evolution', max_iter=1000000, seed=1, weight=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.bounds = [bounds] * feature_num
        self.optimizer = optimizer
        self.coef_ = None
        self.max_iter = max_iter
        self.seed = seed
        self.message = None
        self.weight = weight

    def custom_loss_w(self, params, x, y, weight):
        predictions = np.dot(x, params)
        residuals = (y - predictions) * weight

        Density_mse_loss = np.sum(residuals ** 2)
        L1_norm = np.sum(np.abs(params))  # L1 sparsity regularization

        total_loss = Density_mse_loss + self.alpha * L1_norm  # objective function
        return total_loss

    def fit(self, x, y):
        if self.optimizer == 'differential_evolution':
            result = differential_evolution(self.custom_loss_w, self.bounds, args=(x, y, self.weight), maxiter=self.max_iter, seed=self.seed)
        elif self.optimizer == 'dual_annealing':
            result = dual_annealing(self.custom_loss, self.bounds, args=(x, y), maxiter=self.max_iter)
        elif self.optimizer == 'minimize':
            initial_guess = [0, 0] if self.fit_intercept else [0]
            result = minimize(self.custom_loss, initial_guess, args=(x, y), bounds=self.bounds, maxiter=self.max_iter)
        else:
            raise ValueError("Unsupported optimizer. Use 'differential_evolution' or 'minimize'.")
        self.coef_ = result.x

    def predict(self, x):
        if self.coef_ is None:
            raise ValueError(
                "The model is not trained yet. Call 'fit' with appropriate arguments before using this estimator.")
        return np.dot(x, self.coef_)


dw = DenseWeight(alpha=0.5)
y_pi = y_train[:, 1]
weight = np.array(dw.fit(y_pi)).reshape(-1)
e_obj_w = CustomObj_w(alpha=0.1, feature_num=np.shape(x_train)[1], optimizer='differential_evolution', weight=weight)


# Method based on SVR
svr = SVR(kernel='rbf', C=1e8, epsilon=0.1)

# Method based on Gaussian Process Regression
kernel = DotProduct() + WhiteKernel() + RationalQuadratic()
guasspro = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=5e-8)

# Method based on ANN
ann = MLPRegressor(hidden_layer_sizes=(10, 20, 20, 10), activation='relu', solver='adam', max_iter=1e6)

# Method based on DNN
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def residual_block(x, filters):
    shortcut = x
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Dense(filters)(shortcut)

    x = layers.add([x, shortcut])  # residual connection
    x = layers.ReLU()(x)
    return x


def create_dnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = residual_block(x, 16)
    x = residual_block(x, 32)
    x = residual_block(x, 64)  
    x = residual_block(x, 32)

    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(1)(x)  # 输出层

    model = models.Model(inputs, outputs)
    return model


dnn = create_dnn_model(input_shape)
dnn.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')


# Method based on Random forest
rfr = RandomForestRegressor(criterion='mae', n_estimators=75, min_samples_split=3, min_samples_leaf=3)

# Method based on Xgboost
xgboost = xgb.XGBRegressor(n_estimators=26, max_depth=5, reg_lambda=1, reg_alpha=0.5)
#  ——————————————————————————————————————————————————————————————————————prepare models


#  ——————————————————————————————————————————————————————————————————————select model
model_num = 1
if model_num == 0:
    clf = e_obj
elif model_num == 1:
    clf = e_obj_w
elif model_num == 2:
    clf = svr
elif model_num == 3:
    clf = guasspro
elif model_num == 4:
    clf = ann
elif model_num == 5:
    clf = dnn
elif model_num == 6:
    clf = rfr
elif model_num == 6:
    clf = xgboost
#  ——————————————————————————————————————————————————————————————————————select model


#  ——————————————————————————————————————————————————————————————————————Perturbing the dataset based on Bootstrap method
# Set the number of times Bootstrap sampling is required
n_bootstrap = 30
predictions = []

from sklearn.utils import resample
for i in range(n_bootstrap):
    X_resampled, y_resampled = resample(x_train, y_train, n_samples=len(x_train), random_state=i)

    if model_num == 5:
        clf.fit(X_resampled, y_resampled, epochs=1000, batch_size=64)
    else:
        clf.fit(X_resampled, y_resampled[:, target])

    just_pre = clf.predict(x_test)

    if knowledge_label != 0:
        Diameter = 6.34  # Tianjin 9 use
        true_test_pre = just_pre * y_test[:, 5] * (Diameter ** 2)
        predictions.append(true_test_pre)
    else:
        predictions.append(just_pre)
    print(i)
#  ——————————————————————————————————————————————————————————————————————Perturbing the dataset based on Bootstrap method


#  ——————————————————————————————————————————————————————————————————————save the results
# Convert the predicted results into a DataFrame
predictions_matrix = np.array(predictions).T
predictions_df = pd.DataFrame(predictions_matrix)

predictions_df.to_excel('predictions_output.xlsx', index=False)
print("The predicted results have been saved to 'predictions_output.xlsx'.")
#  ——————————————————————————————————————————————————————————————————————save the results
