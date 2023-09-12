# 1.2% error on MNIST with no tunable hyperparameters
# jitted code to compute the quartic quar_kernel in parallel
# Benjamin Recht. Last update: Oct 23, 2021

import numpy as np
import numba as nb

# compute entrywise M[i,j]^4
@nb.njit(parallel=True)
def fast_quar(M):
    X = np.empty_like(M)
    for x in nb.prange(M.shape[0]):
        for y in nb.prange(M.shape[1]):
            X[x,y] = np.power(M[x,y],4)
    return X

# Normalize all of the rows of M to have 2-norm equal to 1
@nb.njit(parallel=True)
def fast_norm_rows(M):
    X = np.empty_like(M)
    N = np.zeros(M.shape[0])
    for x in nb.prange(M.shape[0]):
        for y in nb.prange(M.shape[1]):
            N[x]+=M[x,y]**2
        N[x] = 1/np.sqrt(N[x])
        for z in nb.prange(M.shape[1]):
            X[x,z] = M[x,z]*N[x]
    return X

def quar_kernel(X,Z):
    return fast_quar(fast_norm_rows(X) @ fast_norm_rows(Z).T)

# load mnist (source: https://s3.amazonaws.com/img-datasets/mnist.npz)
mnist = np.load('mnist.npz')
XTrain = mnist['x_train'].reshape((len(mnist['x_train']),-1)).astype(np.float64)
XTest = mnist['x_test'].reshape((len(mnist['x_test']),-1)).astype(np.float64)

# backpropagate to find the weights using Adam with layer norm and attention:
C = np.linalg.solve(quar_kernel(XTrain,XTrain),np.eye(10)[mnist['y_train']])

# evaluate model on test set and compute the error
labels_pred = np.argmax(quar_kernel(XTest,XTrain) @ C, axis=1)
err = 100.0*np.sum(mnist['y_test']!=labels_pred)/len(mnist['y_test'])
print('Error={:.2f}%.\n'.format(err))
