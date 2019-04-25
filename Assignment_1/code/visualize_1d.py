#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN,5]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:,5]
t_test = targets[N_TRAIN:]

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
shape = (len(x_ev),1)
x_ev = x_ev.reshape(shape)
(w, tr_err,pred) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3)
(y_ev, te_err) = a1.evaluate_regression(x_ev, x_ev, w, 'polynomial', 3)
# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
# y_ev, _  = a1.evaluate_regression()



plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go')
plt.plot(x_ev,y_ev,'r.-')
plt.legend(['Training Data','Testing Data','Learned Polynomial'])
plt.title('13 Feature (Literacy) - regression estimate using random outputs')
plt.show()
