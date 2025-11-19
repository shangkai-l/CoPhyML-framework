"""
Created on March 11 2022
@author: Shangkai
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic
from scipy.optimize import shgo
from scipy import optimize as opt
from scipy.optimize import minimize
from scipy.optimize import dual_annealing, differential_evolution
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import xgboost as xgb
# from denseweight import DenseWeight
import warnings

warnings.filterwarnings("ignore")


def compute_result(true, predict):
    true = np.array(true).reshape(-1, 1)
    predict = np.array(predict).reshape(-1, 1)
    rmse = np.average((true - predict) ** 2) ** (1 / 2)
    diff = np.abs(true - predict)
    mape = np.mean(diff / np.abs(true))
    from sklearn.metrics import r2_score
    r2 = r2_score(true, predict)
    return rmse, mape, r2


#  ——————————————————————————————————————————————————————————————————————准备数据
task_code = 6
# 0：大伙房原始|  1：大伙房pi量|  2：天津9原始|  3：天津3原始|  4：广州3原始|  5：深圳2原始| 6：天津9小pi|  7：天津3小pi| 8：广州3小pi|  9：深圳2小pi
read_path = r"D:\p-excel\博二下工作\计算数据\常用数据_p.xlsx"
# read_path = r"G:\研究生期间资料备份\博三下工作\小论文\小论文2_多种衔接层\绘图\tj9_use.xlsx"

sheet_list = ['大伙房原始', '大伙房pi量', '天津9原始', '天津3原始', '广州3原始', '深圳2原始',  '天津9pi量', '天津3pi量', '广州3pi量', '深圳2pi量']
X_loc_list = [16, 13, 12, 12, 12, 12, 9, 9, 9, 9]
need_trans_list = [1, 6, 7, 8, 9]

data = pd.read_excel(read_path, sheet_name=sheet_list[task_code])

X = np.array(data.iloc[:, 0:X_loc_list[task_code]])
y = np.array(data.iloc[:, X_loc_list[task_code]:])

y_use = y.copy()


Test_size_ = 0.7

x_train, x_test, y_train, y_test = train_test_split(X, y_use, test_size=Test_size_, random_state=0, shuffle=False)

i = 1
target = i  # 预测目标量，0：能耗；1：推力；2：扭矩
#  ——————————————————————————————————————————————————————————————————————准备数据


#  ——————————————————————————————————————————————————————————————————————准备模型
"""
设置基本测试模型，包括最小二乘法、随机森林、系数为正和可负的Lasso
"""
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
        self.initial_guess = initial_guess  # 设定的初始点
        self.x0 = x0

    def custom_loss(self, params, x, y):
        predictions = np.dot(x, params)
        residuals = y - predictions  # 坑死了，报错先看需不需要转换成（-1，1），可能广播机制就会变成（n,n）的数组，导致寻优结果很离谱
        # ε不敏感损失
        epsilon_loss = np.sum(np.maximum(0, np.abs(residuals) - self.epsilon))
        epsilon_mse_loss = np.sum(np.maximum(0, residuals ** 2 - (self.epsilon) ** 2))

        # # MSE损失
        MSE_loss = np.sum(residuals ** 2)

        # L1正则化
        # l1_regularization = self.alpha * (
        #             np.abs(intercept) + np.abs(slope)) if self.fit_intercept else self.alpha * np.abs(slope)
        L1_norm = np.sum(np.abs(params))
        L2_norm = np.sum(params ** 2)
        # 总损失
        # total_loss = epsilon_loss + self.alpha * L1_norm
        # total_loss = MSE_loss + self.alpha * L2_norm
        total_loss = epsilon_mse_loss + self.alpha * L1_norm
        # print(1)
        return total_loss

    def fit(self, x, y):
        if self.optimizer == 'differential_evolution':
            if self.initial_guess is not None:

                result = differential_evolution(self.custom_loss, self.bounds, args=(x, y),
                                                maxiter=self.max_iter, seed=self.seed, tol=1e-10,
                                                init=1)  # 传入自定义种群
            else:
                result = differential_evolution(self.custom_loss, self.bounds, args=(x, y),
                                                maxiter=self.max_iter, seed=self.seed, tol=1e-10)
            result = differential_evolution(self.custom_loss, self.bounds, args=(x, y), maxiter=self.max_iter, seed=self.seed, tol=1e-10)
            # result = differential_evolution(self.custom_loss_w, self.bounds, args=(x, y, self.weight), maxiter=self.max_iter, seed=self.seed)
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



"""
设置基本测试模型，包括最小二乘法、随机森林、系数为正和可负的Lasso
"""
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
        # ε不敏感损失
        epsilon_loss = np.sum(np.maximum(0, np.abs(residuals) - self.epsilon))

        # # MSE损失
        MSE_loss = np.sum(residuals ** 2)

        # L1正则化
        # l1_regularization = self.alpha * (
        #             np.abs(intercept) + np.abs(slope)) if self.fit_intercept else self.alpha * np.abs(slope)
        L1_norm = np.sum(np.abs(params))
        L2_norm = np.sum(params ** 2)
        # 总损失
        total_loss = MSE_loss + self.alpha * L1_norm
        return total_loss

    def fit(self, x, y):
        if self.optimizer == 'differential_evolution':
            # result = differential_evolution(self.custom_loss, self.bounds, args=(x, y), maxiter=self.max_iter, seed=self.seed, tol=1e-10)
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



lr = LinearRegression(fit_intercept=False)  # 最小二乘法
rfr = RandomForestRegressor(max_depth=5, random_state=0, criterion='mae', n_estimators=100, min_samples_split=3, min_samples_leaf=3)  # 随机森林
lasso_1 = linear_model.Lasso(alpha=0.2, fit_intercept=False, positive=False,
                         random_state=0, max_iter=500000, selection='random', tol=1e-20)
# 0.00631   0.00108
lasso_2 = linear_model.Lasso(alpha=0.00131, fit_intercept=False, positive=False,
                         random_state=0, max_iter=500000, selection='random', tol=1e-20)
ridge = linear_model.Ridge(alpha=4.7e-3, fit_intercept=False)
ann = MLPRegressor(hidden_layer_sizes=(10, 20, 20, 10), activation='relu', solver='adam', max_iter=100000, tol=1e-20, learning_rate_init=1e-4, n_iter_no_change=300)
svr = SVR(kernel='rbf', C=1e8)
knn = KNeighborsRegressor(weights="uniform")
kernel = DotProduct() + WhiteKernel() + RationalQuadratic()
guasspro = GaussianProcessRegressor(kernel=kernel, random_state =1, normalize_y=True, alpha=5e-8)
decision_tree = tree.DecisionTreeRegressor(max_depth=20, criterion='mse')
xgboost = xgb.XGBRegressor(n_estimators=26, max_depth=5, reg_alpha=0.5)
e_loss = SVR(kernel='linear', epsilon=10)


# # BIC选参数
# from sklearn import linear_model
# reg = linear_model.LassoLarsIC(criterion='bic', fit_intercept=False, noise_variance=700)
# reg.fit(x_train, y_train[:, target])
# print(reg.alpha_)
#  ——————————————————————————————————————————————————————————————————————准备模型
coef11 = []
base_coef = np.array([-3.71036600e+01, -1.50522729e-01,  6.28273252e+02,  6.91220697e+02,
       -5.75695448e-01, -1.03789848e+01, -1.33385486e+02, -2.02570137e+01,
        2.91310460e+00])
# base_coef = np.array([0.49813184, 0.49999219, 0.53151673, 0.52539678, 0.49998962, 0.49987863,
#  0.49646764, 0.49933386, 0.50012805])
for seed in range(0, 1):
    # tj9: 6.8,0.055,  L2 6, 1e-5; sz2: L2 0.25, 0.1, L1 0.5, 10
    # e_obj = CustomObj(epsilon=6.8, alpha=5.5e-2, feature_num=np.shape(x_train)[1], optimizer='differential_evolution', seed=seed, max_iter=10, initial_guess=base_coef)
    e_obj = CustomObj(epsilon=6.8, alpha=0.055, feature_num=np.shape(x_train)[1], optimizer='differential_evolution', seed=seed, initial_guess=None)

    # dw = DenseWeight(alpha=0.15)
    # y_pi = y_train[:, 1]
    # weight = np.array(dw.fit(y_pi))
    weight = np.array(pd.read_excel('./w.xlsx')).reshape(-1)
    e_obj_w = CustomObj_w(alpha=0.1, feature_num=np.shape(x_train)[1], optimizer='differential_evolution', seed=seed, weight=weight)
    model_num = 1
    if model_num == 0:
        clf = e_obj_w
    elif model_num == 1:
        clf = e_obj
    elif model_num == 2:
        clf = ann
    else:
        clf = lasso_2

    # scaler_x = MinMaxScaler()
    # x_train = scaler_x.fit_transform(x_train)
    # x_test = scaler_x.transform(x_test)
    # X = scaler_x.transform(X)
    clf.fit(x_train, y_train[:, target])
    # clf.fit(X, y_use[:, target])
    # x_test = x_train
    # y_test = y_train
    just_pre = clf.predict(x_test)
    # just_pre = x_test @ np.array([1.42, -0.114, 0, 0.617, 101, 0])
    just_pre_all = clf.predict(X)

    """
    结果评估与验证，主要包括：
    1. 转换前的R2
    2. 转换后的R2
    """

    if task_code in need_trans_list:
        # ——————————————————————————————————————————————————————————————大/小pi量数据使用
        # 转换后的R2
        if task_code == 1:
            Diameter = 8.03  # 大伙房使用
        else:
            Diameter = 6.34  # 天津9/3 使用

        if target == 0:
            if task_code == 1:
                true_test_pre = just_pre * y_test[:, 5] * 1000 * (Diameter ** 3) * x_test[:, 1]   # 大伙房能耗使用
                true_pre = just_pre_all * y_use[:, 5] * 1000 * (Diameter ** 3) * X[:, 1]  # 大伙房能耗使用
            else:
                true_test_pre = just_pre * y_test[:, 5] * (Diameter ** 3)  # 天津9/3能耗使用
                true_pre = just_pre_all * y_use[:, 5] * (Diameter ** 3)  # 天津9/3能耗使用
        elif target == 1:
            true_test_pre = just_pre * y_test[:, 5] * (Diameter ** 2)   # 推力使用
            true_pre = just_pre_all * y_use[:, 5] * (Diameter ** 2)  # 推力使用
        else:
            true_test_pre = just_pre * y_test[:, 5] * (Diameter ** 3)  # 扭矩使用
            true_pre = just_pre_all * y_use[:, 5] * (Diameter ** 3)  # 推力使用

        # print('Predict (%s) using model (%s)' % (target_list[i], model_list[model_num]))
        print(compute_result(y_test[:, 6 + target], true_test_pre), '____', seed)
        pic_pred = true_pre
        coef11.append(compute_result(y_test[:, 6 + target], true_test_pre))
    else:
        # ——————————————————————————————————————————————————————————————原始数据使用
        # R2
        # print('Predict (%s) using model (%s)' % (target_list[i], model_list[model_num]))
        print(compute_result(y_test[:, target], just_pre), '____', seed)
        pic_pred = just_pre_all
        # coef11.append(compute_result(y_test[:, 6 + target], true_test_pre))

# pd.DataFrame(np.array(coef11)).to_excel(r'D:\p-excel\博二下工作\计算数据\计算数据/result_w_L2_0.5.xlsx')
plt.scatter(np.arange(0, len(y_use[:, target])), y_use[:, -(3 - target)], label='Measured data', s=5)
plt.scatter(np.arange(0, len(y_use[:, target])), pic_pred, label='Pred data', s=5)
plt.plot([len(y_use[:, target]) * (1 - Test_size_)] * 2, [y_use[:, -(3 - target)].min()*0.6, y_use[:, -(3 - target)].max()*1.2], color='white')
plt.xlabel('Ring num')
plt.ylabel('F(kN)')
plt.legend()
plt.show()

# # 存结果

# merged_array2 = np.column_stack((y_use[:, -(3 - target)], pic_pred, y_use[:, target], just_pre_all))
# pd.DataFrame(merged_array2).to_excel(r'D:\p-excel\博二下工作\计算数据\计算数据/pred_pi.xlsx')

"""
超参调试
"""
# # alpha_list = [1e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 1e-1, 5e-1, 1, 2, 3, 4, 5, 10]
# alpha_list = list(np.arange(1, 10, 0.05)*1e-2)
# Diameter = 6.34
# mape_list = []
# zero_list = []
#
# for alpha in alpha_list:
#     rgs = linear_model.Lasso(alpha=alpha, fit_intercept=False, positive=False,
#                                  normalize=False, random_state=0, max_iter=500000, selection='random', tol=1e-20)
#     rgs.fit(x_train, y_train[:, target])
#     just_pre = rgs.predict(x_test)
#     true_test_pre = just_pre * y_test[:, 5] * (Diameter ** 2)   # 推力使用
#     _, m, _ = compute_result(y_test[:, target], just_pre)
#     mape_list.append(m)
#     zero_list.append(np.count_nonzero(rgs.coef_ == 0))
#
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(np.log10(np.array(alpha_list)), mape_list)
#
# plt.figure()
# plt.scatter(np.log10(np.array(alpha_list)), zero_list)


# ——————————————————————————牵引稀疏展示
# def e_loss(params, x, y, e):
#     predictions = np.dot(x, params)
#     residuals = y - predictions  # 坑死了，报错先看需不需要转换成（-1，1），可能广播机制就会变成（n,n）的数组，导致寻优结果很离谱
#     # ε不敏感损失
#     epsilon_mse_loss_array = np.maximum(0, residuals ** 2 - (e) ** 2)
#     epsilon_mae_loss_array = np.maximum(0, np.abs(residuals) - e)
#     # return np.count_nonzero(epsilon_mse_loss_array)
#     return epsilon_mae_loss_array


# write_list = []
# for e in range(0, 12, 3):
#     # write_list.append((e/5, e_loss(clf.coef_, x_train, y_train[:, target], e/5)))
#
#     write_list.append(e_loss(clf.coef_, x_train, y_train[:, target], e))
#
# pd.DataFrame(write_list).transpose().to_excel(r'E:\研究生期间资料备份\博三上工作\计算结果\20240906\diff_e_mae_loss.xlsx')

# 存结果
# coef_array = e_obj.coef_
# custom_name = "no_iteg_260"  # 你的自定义名称
# file_path = r"G:\研究生期间资料备份\博三下工作\小论文\小论文2_多种衔接层\绘图\迭代方向\系数.xlsx"  # 你的Excel文件路径
# sheet_name = "Sheet1"  # 你要写入的Excel表单
#
# # 读取已有的Excel
# try:
#     df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')  # 先读取已有数据
# except FileNotFoundError:
#     df = pd.DataFrame()  # 如果文件不存在，则创建一个空DataFrame
#
# # 创建新的行数据
# new_row = [custom_name] + coef_array.tolist()
#
# # 追加到DataFrame
# df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], axis=0, ignore_index=True)
#
# # 写入Excel，保持原有数据，不覆盖
# # df.to_excel(file_path, sheet_name=sheet_name, index=False, header=True, engine='openpyxl')
#
# print("数据成功追加到Excel文件！")

