#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = {}
test_err = {}

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py






lamda_val = [0,.01,.1,1,10,100,1000,10000]
#lamda_val = [0]
avg_lambda_err = []

# Produce a plot of results.
for i in lamda_val:
    er = []
    for j in range(10,101,10):
        x_val_set = x_train[j-10:j,:]
        val_1 = x_train[0:j-10,:]
        val_2 = x_train[j:100,:]
        x_train_set = np.vstack((val_1, val_2))
        t_val_set = t_train[j-10:j,:]
        val_1 = t_train[0:j-10,:]
        val_2 = t_train[j:100,:]
        t_train_set = np.vstack((val_1, val_2))
        (w, tr_err,pred) = a1.linear_regression(x_train_set, t_train_set, 'polynomial', i, 2)
        (y_ev, te_err) = a1.evaluate_regression(x_val_set, t_val_set, w, 'polynomial', 2)
        er.append(te_err)
    avg_lambda_err.append(np.mean(er))
# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
# y_ev, _  = a1.evaluate_regression()

plt.semilogx(lamda_val, avg_lambda_err)
plt.ylabel('Average Vaildation Set Error')
plt.title('Regularized Polynomial Regression 10 Fold')
plt.xlabel('Lambda on log scale')
plt.show()
