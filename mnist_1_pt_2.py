# 1.2% error on MNIST with no tunable hyperparameters
# Benjamin Recht. Last update: Oct 23, 2021

import numpy as np

# helper functions for computing kernel.
# normalizes the Euclidean norm of the rows of X:
normalize = lambda X : np.power(np.sum(X**2,axis=1)[:,None],-0.5)*X
# computes quartic kernel of normalized X and Z:
quar_kernel = lambda X,Z : np.power(normalize(X) @ normalize(Z).T,4)

################################################################################
# load mnist (source: https://s3.amazonaws.com/img-datasets/mnist.npz)
mnist = np.load('mnist.npz')
XTrain = mnist['x_train'].reshape((len(mnist['x_train']),-1)).astype(np.float64)
XTest = mnist['x_test'].reshape((len(mnist['x_test']),-1)).astype(np.float64)

# backpropagate to find the weights using Adam with layer norm and attention:
C = np.linalg.solve(quar_kernel(XTrain,XTrain),np.eye(10)[mnist['y_train']])

# evaluate model on test set and compute the error
labels_pred = np.argmax(quar_kernel(XTest,XTrain) @ C, axis=1)
err = 100.0*np.sum(mnist['y_test']!=labels_pred)/len(mnist['y_test'])
print('Error={:.2f}%.'.format(err))
