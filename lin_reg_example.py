from os import X_OK

from numpy.core.fromnumeric import transpose
from pandas.core.indexes.base import ensure_index
from d_opt import d_opt
import numpy as np
import pandas as pd
import time

data = pd.read_table("processed_lin_reg.txt")
X = data[:, 3:6]
y = data[:, 7]

X = np.transpose(X)
dimensions = np.shape(X)
n = dimensions[0]  # number of samples
d = dimensions[1]  # number of features
# confused by the following;
# X = [X; ones(1, n)];

Budgets = np.arange(1, 83)
# Once, d_opt works, this should work
lambs, fvals = d_opt(X, Budgets)


###########
# Pipage, explanation in book.
round_lamb = np.round(round_lambs)
############
X = np.transpose(X)

beta_all_1 = np.add(np.identity(d), np.matmul(np.transpose(X), X))
beta_all_2 = np.matmul(np, transpose(X), y)

beta_all = np.divide(beta_all_1, beta_all_2)

dims = np.shape(round_lambs)
m = dims[1]  # features
betas = np.zeros(n, m)
t = time.time()
for i in 1:
    len(m):
    lambd = round_lambs[:, i]
    ind = np.argwhere(lambd)
    Xtemp = X[ind, :]
    ytemp = y[ind, :]

    # function call for betas
    dum1 = np.add(np.identity(d), np.matmul(np.transpose(Xtemp), Xtemp))
    dum2 = np.matmul(np.transpose(Xtemp), ytemp)
    betas(:, i) = np.divide(dum1, dum2)
# end
elapsed = time.time() - t

[betas, beta_all]
