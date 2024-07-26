"""
    Function：Test the tch function
    Author：Tongren Xu (xutr@bnu.edu.cn), Gangqiang Zhang(zhanggq@mail.bnu.edu.cn) and Xinlei He (hxlbsd@mail.bnu.edu.cn) 
    Version：Python 3.6.8
    Reference: Ferreira, V.G., Montecino, H.D.C., Yakubu, C.I., Heck, B., 2016. Uncertainties of the Gravity Recovery and Climate Experiment time-variable gravity-field solutions based on three-cornered hat method. J. Appl. Remote Sens 10, 015015. https://doi.org/10.1117/1.JRS.10.015015
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def tch(x):
    # objective function
    def fun_object(r, S):
        N = S.shape[0]
        f = 0.0
        K2 = np.linalg.det(S) ** (2 / N)
        for i in range(N):
            f = f + r[i] ** 2  # why?
            for j in range(i + 1, N):
                f = f + (S[i, j] - r[N] + r[i] + r[j]) ** 2
        return f / K2

    # constraint conditions
    def fun_constraint(r, S):
        N = S.shape[0]
        u = np.full((N,), 1.0)
        K = np.linalg.det(S) ** (1 / N)
        f = (r[N] - (r[:-1] - r[N] * u).dot(np.linalg.inv(S)).dot(r[:-1] - r[N] * u)) / K
        return f

    def tch(x):
        M, N = x.shape  # M: samples; N:variables
        N_ref = N  # the last column as the reference dataset
        y_list = []
        for i in range(N):
            if i == N_ref - 1:
                pass
            else:
                y_list.append(x[:, i] - x[:, N_ref - 1])
        Y = np.vstack(y_list).T  # size = M*N-1
        S = np.cov(Y.T)  # it is different from the operation of the matlab cov. size:(N-1 * N-1)
        u = np.full((1, N - 1), 1.0)
        R = np.zeros((N, N))
        R[N - 1, N - 1] = 1 / (2 * u.dot(np.linalg.inv(S)).dot(u.T))
        X0 = R[:, N - 1]
        # According to the initial conditions, constraint conditions, and objective function of the iteration, R(:,N-1) is calculated
        res = minimize(lambda r: fun_object(r, S), X0, method='SLSQP', constraints={'type': 'ineq', 'fun': lambda r: fun_constraint(r, S)})
        R[:, N - 1] = res.x
        R[0:-1, 0:-1] = S - R[N - 1, N - 1] * (u.T.dot(u)) + u.T.dot(R[:-1, N - 1:N].T) + R[:-1, N - 1:N].dot(u)

        # Adjust the order
        R_list = [R[ii, ii] for ii in range(N)]
        R_list_end = R_list[-1]
        R_list.pop()
        R_list.insert(N_ref - 1, R_list_end)

        std = np.round([np.sqrt(R_list[ii]) for ii in range(N)], 3)
        std_xd = np.round([np.sqrt(R_list[ii]) / np.nanmean(abs(x[:, ii])) for ii in range(N)], 3)
        std_list, std_xd_list = list(std), list(std_xd)
        print("std:{}\nstd_xd:{}".format(std_list, std_xd_list))

        return np.array(std_list), np.array(std_xd_list), res.success

    return tch(x)


if __name__ == '__main__':
    # test
    data = pd.read_csv(r'data.csv', index_col=0)
    x = data.iloc[:].values
    a = tch(x)
