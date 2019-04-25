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
for degree in range(1,7):
    (w, tr_err,pred) = a1.linear_regression(x_train, t_train, 'polynomial', 0, degree)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree)
    train_err[degree] = tr_err.item(0)
    test_err[degree] = te_err.item(0)


# Produce a plot of results.
plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
