#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,3]
x_test = x[N_TRAIN:,3]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = {}
test_err = {}

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

# Produce a plot of results.

(w, tr_err, pred) = a1.linear_regression(x_train, t_train, 'ReLU', 0, 2)
(y_ev, te_err) = a1.evaluate_regression(x_test, t_test, w, 'ReLU', 2)
temp = np.hstack((x_train,pred))
temp1 = np.asarray(temp)
temp1 = temp1[temp1[:,0].argsort()]
# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
# y_ev, _  = a1.evaluate_regression()

print ('Training error for this model is :', tr_err)
print ('Testing error for the model is :', te_err)

plt.plot(x_train,t_train,'bo')
plt.plot(temp1[:,0],temp1[:,1],'r.-')

plt.title('A visualization of ReLU regression')
plt.show()
#plt.show()
