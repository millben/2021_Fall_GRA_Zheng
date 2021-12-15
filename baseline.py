import numpy as np
from numpy.core.numeric import argwhere
from numpy.lib.shape_base import expand_dims
from numpy.linalg import norm
import pandas as pd

lars = pd.read_csv("lars.txt", header=None, delim_whitespace=True)
lars


B = np.arange(50, 221, 10)
nB = len(B)
nB
dum = len(lars.columns)
X = lars.iloc[:, 0:dum-1]
y = lars.iloc[:, 0:dum]
X


n = lars.shape[0]
epsilon = 0.9
muX = np.mean(X)
sigmaX_norm = np.linalg.norm(np.std(X))

maxiter = 10
filtered = np.zeros((n, lars.shape[1]+1, maxiter))
num_filtered_iter = np.zeros((maxiter))

for i in range(maxiter):

    # construct a ball of radius sigmaX around muX and delete points inside the ball
    # havent tested yet, but in matlab, broadcasting doesnt happen automatically, however, with numpy, it should happen automatically (thus no need for matlab function bsxfun)
    x_minus_mu = X - muX
    # took out np.sum at beginning
    rownorm = np.sqrt(np.sum(np.transpose(x_minus_mu)**2))
    # distance from all points from the mean // why transpose tho? this does nothing
    rownorm = np.transpose(rownorm)

    # now we will pick the samples outside the ball

    for j in range(len(rownorm)):
        if rownorm[j] < epsilon*sigmaX_norm:
            rownorm[j] = 0
        else:
            rownorm[j] = 1

    num_filtered_iter[i] = len(np.unique(np.where(rownorm == 0)))
    rownorm

    # rownorm now consists of subjects that are selected to the next round

    # The translation from matlab could be either of these lines below
    filtered[1:rownorm.shape(0) - np.argwhere(rownorm != 0).shape(0), :, i] = [
        X[np.argwhere(rownorm == 0), :], y[np.argwhere(rownorm == 0), :]]
    filtered[i, 1:rownorm.shape(0) - np.argwhere(rownorm != 0).shape(0), ] = [
        X[np.argwhere(rownorm == 0), ], y[np.argwhere(rownorm == 0), ]]
    ##########

    # update X, muX, sigmaX_norm
    X = np.where(rownorm != 0)
    y = np.where(rownorm != 0)

    Xtest = np.asarray(X)
    Xtest
    Xtest.shape

    # This code chunck translated does not make sense. Where does Y come from?
    # why keep track of the specific filtered values??
    # if Xtest.shape(0) == 1:
    # This line below as well!
    # Y(1, :, iter+1) = [X,y]
    #y[i+1,1,] = [Xtest,y]
    # break
    # elif Xtest.shape(0) == 0:
    # break
    # else:
    muX = np.mean(Xtest)
    sigmaX_norm = np.linalg.norm(np.std(Xtest))
#####################################


num_to_pick = np.zeros((len(num_filtered_iter), nB))

for i in range(nB):
    num_to_pick[:, i] = np.round((num_filtered_iter/n) * B[i])

num_to_pick
np.shape(num_to_pick)
np.dtype(num_to_pick)
