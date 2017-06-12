#!/usr/bin/env python3
__author__ = 'Evgeniy.Tumanov'

import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import time
import sys
from sklearn.linear_model import LinearRegression

GRID_SIZE = 10
n = 10
t = sys.argv[1]


class kalman:

    def kalman_rec(self, y_t, H_t, xi_t_tm1, P_t_tm1, F_tp1, Q_t, R_t):
        # for debug: may be useful ->
        # -> print(y_t.ndim, H_t.ndim, F_tp1.ndim, Q_t.ndim, R_t.ndim, xi_t_tm1.ndim, P_t_tm1.ndim)

        y_t_hat = T.dot(H_t.T, xi_t_tm1)
        eps_t = y_t - y_t_hat
        pseudo_inv = T.nlinalg.matrix_inverse(T.dot(H_t.T, T.dot(P_t_tm1, H_t)) + R_t)
        invi =  T.dot(P_t_tm1, T.dot(H_t, pseudo_inv))
        xi_t_t = xi_t_tm1 + T.dot(invi, eps_t)
        xi_tp1_t = T.dot(F_tp1, xi_t_t)
        P_t_t = P_t_tm1 - T.dot(invi, T.dot(H_t.T, P_t_tm1))
        P_tp1_t = T.dot(F_tp1, T.dot(P_t_t, F_tp1.T)) + Q_t

        return [xi_tp1_t, P_tp1_t, y_t_hat, xi_t_t, T.sqr(eps_t)]

    def __init__(self, Y, X):
        self.Y, self.X = Y[:, :, np.newaxis], X
        self.n = X.shape[1]
        y, H = T.tensor3(), T.tensor3()
        F, Q, R = T.matrix(), T.matrix(), T.matrix()
        sequences=[dict(input=y, taps=[0]), dict(input=H, taps=[0])]
        xi_1_0 = T.matrix()
        P_1_0 = T.matrix()
        (values, updates) = theano.scan(fn=self.kalman_rec,
                                          sequences=sequences,
                                          outputs_info=[xi_1_0, P_1_0, None, None, None],
                                          non_sequences=[F, Q, R],
                                          strict=True)
        self.kalman_filter = theano.function([y, H, xi_1_0, P_1_0, F, Q, R], values, updates=updates, allow_input_downcast=True)

    def get_G_mat(self, x):
        G = np.zeros((self.n, 2*self.n), dtype='float32')
        source = np.hstack([x.reshape(-1, 1), np.ones((x.shape[0], 1))])
        for i in range(self.n):
            G[i, i* 2] = source[i, 0]
            G[i, i* 2+1] = source[i, 1]
        return G

    def filter_out(self, Q, R):
        # init all variables need
        H_val = np.array([self.get_G_mat(self.X[t]).T for t in range(self.X.shape[0])])
        F_val = np.diag(np.ones(2*self.n, dtype='float32'))

        xi_1_0_val = np.zeros((2*self.n, 1))
        P_1_0_val = np.diag(np.ones(2*self.n))
        # for debug: may be useful ->
        # -> print('Shapes:\n')
        # -> print(' y : {}\n H : {}\n xi_1_0 : {}\n P_1_0 : {}\n F : {}\n Q : {}\n R : {}\n'.format(*map(\
        #                            lambda d: d.shape, [self.Y, H_val, xi_1_0_val, P_1_0_val, F_val, Q, R])))
        start_time = time.time()
        output = self.kalman_filter(self.Y, H_val, xi_1_0_val, P_1_0_val, F_val, Q, R)
        # for debug: may be useful ->
        # -> print('Eval time: {} s.'.format(time.time() - start_time))
        return output


def kalman_apply(data_slice, pairs, left, right):

    pairs_subset = np.array(pairs.iloc[left:right])
    Y = np.array(data_slice[pairs_subset[:, 0]])
    X = np.array(data_slice[pairs_subset[:, 1]])

    sigmas = Y.std(axis=0)
    optim_grid = np.hstack([np.linspace(0., 1., GRID_SIZE)[1:].reshape(-1, 1) for i in range(n)]) * sigmas
    beta_cells = []
    for i in range(n):
        y, x = Y[:, i], X[:, i]
        beta_cells.append(LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1)).coef_.ravel()[0] / np.sqrt(x.shape[0]))
        model = kalman(Y, X)
        min_errs = np.array([np.infty] * n)
        opt_node = {}
        level = {}
        beta = {}

    for k, node in enumerate(optim_grid):
        R = np.diag(node)
        trace = np.zeros(2*n)
        trace[::2] = node * 0.1
        trace[slice(1, None, 2)] = np.array(beta_cells)
        Q = np.diag(trace)
        output = model.filter_out(Q, R)

        errs = output[-1].mean(axis=0).ravel()
        for i, (cur_err, min_err) in enumerate(zip(errs, min_errs)):
            if cur_err < min_err:
                min_errs[i] = cur_err
                opt_node[i] = k
                #print(output[-2].shape)
                beta[i] = output[-2][:, 2*i]
                level[i] = output[-2][:, 2*i+1]

    for i, (symbol_y, symbol_x) in enumerate(zip(pairs_subset[:, 0], pairs_subset[:, 1])):
        df = pd.DataFrame(np.hstack([beta[i], level[i]]), index=data_slice.index)
        df.columns = ['beta', 'level']
        df.to_csv('D:/data/storage/xis/xi_{}_{}_{}.csv'.format(symbol_y, symbol_x, t))

def main():
    path = 'D:/data/storage/slices/{}.csv'
    data_slice = pd.read_csv(path.format('data_slice_{}'.format(t)), parse_dates=['Date_Time'], index_col='Date_Time')
    pairs = pd.read_csv(path.format('pairs_{}'.format(t)), index_col='id')
    i_stamps = list(range(0, pairs.shape[0], n))
    i_stamps.append(pairs.shape[0])
    i_stamps = np.unique(i_stamps)
    for i, stamp in enumerate(i_stamps[:-1]):
        kalman_apply(data_slice, pairs, stamp, i_stamps[i+1])


if __name__ == "__main__":
    # execute only if run as a script
    main()