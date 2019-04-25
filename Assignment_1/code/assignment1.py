"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_',encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)

    return (x - mvec)/stdvec
    #pca = PCA(whiten=True)
    #return (pca.fit_transform(x))

def weight_cal(phi, t, reg_lambda):
    phi_transpose = np.transpose(phi)
    phi_t_phi = np.dot(phi_transpose,phi)
    lamda_I = np.dot(np.identity(phi_t_phi.shape[0]),reg_lambda)
    intermediate_sum = lamda_I + phi_t_phi
    inverse_reg = np.linalg.pinv(intermediate_sum)
    phi_t_t = np.dot(inverse_reg,phi_transpose)
    w = np.dot(phi_t_t, t)
    return w

def linear_regression(x, t, basis, reg_lambda, degree):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    # e.g. phi = design_matrix(x,basis, degree)
    phi = design_matrix(x,basis, degree)
    #phi = np.c_[ np.ones(len(phi)), phi]

    # TO DO:: Compute coefficients using phi matrix

    w = weight_cal(phi, t, reg_lambda)
    #w = np.dot(np.linalg.pinv(phi),t)
    pred = np.dot(phi,w)
    # Measure root mean squared error on training data.
    su = pred-t
    sum_sq = np.sqrt(np.mean(np.power(su,2)))
    train_err = sum_sq

    return (w, train_err, pred)

def design_matrix(x, basis, degree):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions
    if basis == 'polynomial':
        phi = np.hstack((np.ones(x.shape[0])[:,np.newaxis],x))
        for j in range(2,degree+1):
            phi = np.hstack((phi,np.power(x,j)))
    elif basis == 'ReLU':
        z = np.hstack((np.ones(x.shape[0])[:,np.newaxis],x))
        c = (-1)*z+5000
        phi = np.maximum(c,0)

    else:
        assert(False), 'Unknown basis %s' % basis

    return phi

def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset
      """
    #phi = np.hstack((np.ones(x.shape[0])[:,np.newaxis],x))
  	# TO DO:: Compute t_est and err
    if (basis == 'polynomial'):
        test_phi = design_matrix(x,basis, degree)
    else:
        test_phi = np.hstack((np.ones(x.shape[0])[:,np.newaxis],x))
    #test_phi = np.c_[ np.ones(len(test_phi)), test_phi]
    t_est = np.dot(test_phi,w)
    su = t_est-t
    err = np.sqrt(np.mean(np.power(su,2)))

    return (t_est, err)
