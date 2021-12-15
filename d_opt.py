import numpy as np
from numpy.core.numeric import identity
from numpy.lib.shape_base import expand_dims
import pandas as pd
import time

# This function selects budget number of samples. X is the data matrix with
#  rows as features and columns as samples. This assumes that the constant
#  coefficient is included in the matrix, that is, the last row is all 1s.


def d_opt(X, Budgets):

    dimensions = np.shape(X)
    d = dimensions[0]    # number of samples
    n = dimensions[1]    # number of features

    # normalize data so that the 2-norm of each sample is atmost 1.
    maxNorm = np.max(np.sqrt(np.abs(X)**2))
    X = X/maxNorm
    c = np.ones(n)  # np.ones(n) makes an array of size n of all ones

    # Maximum iterations for each budget
    maxiter = 50
    # This constant makes sure that the logdet is not infinity.
    scaling = 10500

    lamb = np.zeroes(n)  # same as np.ones but with zeroes
    idn = np.identity(n)  # returns an n by n identity matrix
    idd = np.identity(d)  # ''

    # returns a tuple (for ex. a 2x2 matrix would return (2,2) with 0 index as rows and 1 index as columns)
    budget_size = np.shape(Budgets)

    # store function values for plotting purposes
    fval = np.zeros((maxiter, budget_size[0]))
    # store all lambdas
    lambs = np.zeros(n, budget_size[0])
    b = 1
    # Keeps track of runtime
    t = time.time()
    # in the matlab files it seems to only loop through the number of columns present in the Budgets matrix, thus we loop through number of columns
    for i in budget_size[1]:
        lamb = np.zeroes(n)
        print("Budget = ", i)
        for j in maxiter:
            if np.mod(j, 10) == 0:
                print("Iteration = ", j)
            # end -- author has end outside of this if statement? would end the for loop at every iteration before below chunk is ran??
            # Compute gradient

            Xhat = np.matmul(X, np.diag(lamb))
            dum1 = np.add(idn, np.transpose(X))
            dum2 = np.matmul(dum1, Xhat)
            dum3 = np.matmul(np.linalg.inv(dum2), np.transpose(X))
            temp = np.subtract(idd, dum3)
            grad_dummy1 = np.matmul(np.transpose(X), temp)
            grad_dummy2 = np.matmul(grad_dummy1, X)
            grad = -np.diag(grad_dummy2)

            # Solve subproblem ??? CPLEX
            # lambdabar = cplexlp(grad,[],[],ones(n,1)',B,zeros(n,1),ones(n,1));
            ###########

            if (np.transpose(grad) * (lambdabar - lambs) == 0):
                break
            # end
            eta = 2/(i+2)
            lambs = lambs + eta * (lambdabar - lambs)

            fval(i, b) = np.log(np.linalg.det(np.identity(n) + np.matmul(np.matmul(np.transpose(X), X), np.diag(lambs))))

        # end
        lambs[:, b] = lambs
        b = b+1
    # end
    elapsed = time.time() - t

    # end
