#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = []
test_err = []
feat = []

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
for col in range(0,8):
    temp = x_train[:,col]
    temp1 = x_test[:,col]
    (w, tr_err,pred) = a1.linear_regression(temp,t_train,'polynomial',0, 3)
    (t_est, te_err) = a1.evaluate_regression(temp1, t_test, w, 'polynomial', 3)
    train_err.append(tr_err.item(0))
    test_err.append(te_err.item(0))
    feat.append(8+col)

ind = np.arange(8)
width = 0.35     # the width of the bars: can also be len(x) sequence

plt.bar(ind,  train_err, width, color='green')
plt.bar(ind + width,  test_err, width, color='red')

# Produce a plot of results.

plt.ylabel('RMS')
plt.xticks(np.arange(8), feat)
plt.legend(['Training error','Test error',])
plt.title('Feature Number vs RMS, no regularization')
plt.xlabel('Feature Number')
plt.show()
