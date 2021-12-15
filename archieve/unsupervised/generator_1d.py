# import matplotlib.pyplot as plt
import functools
import os
import platform
import shutil
from typing import Callable, Tuple, Union

import ml_collections
import numpy as np
from ml_collections import ConfigDict
from numpy import linalg as LA
from scipy import integrate, linalg
from tqdm import tqdm


def qnwlege1(n, a, b):
    mu, omega = np.polynomial.legendre.leggauss(n)
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)
    mu = mu * xl + xm
    omega = xl * omega

    return mu.astype(np.float32), omega.astype(np.float32)


def eigens(x_i, mu, omega, sigma_T, varepsilon, sigma_a, M):
    x_i -= 1
    D, V = LA.eig(
        (1 / mu).reshape(-1, 1)
        * (
            (sigma_T[x_i] - varepsilon[x_i] ** 2 * sigma_a[x_i])
            * np.ones((2 * M, 2 * M))
            @ np.diag(omega)
            - sigma_T[x_i] * np.eye(2 * M, 2 * M)
        )
    )
    # sort eigenvalues
    D_arg = np.argsort(D)
    V = (V.transpose()[D_arg]).transpose()
    D = D[D_arg]

    if sigma_a[x_i] < 1e-14:
        D[M - 1] = 0
        D[M] = 0
        V[:, M - 1] = np.ones(2 * M)
        V[:, M] = np.ones(2 * M)
    return V, D


def sss(x, x_i, varepsilon, h, sigma_T, xl, mu, M, sigma_a, omega):
    matrix = np.zeros([2 * M, 2 * M])
    V, D = eigens(
        x_i,
        mu=mu,
        omega=omega,
        sigma_T=sigma_T,
        varepsilon=varepsilon,
        sigma_a=sigma_a,
        M=M,
    )

    x_0 = (x_i - 0.5) * h + xl

    matrix = (
        np.exp((D * (x - x_0 - np.sign(D) * 0.5 * h)) / varepsilon[x_i - 1])
        * V
    )
    if sigma_a[x_i - 1] <= 1e-8:
        matrix[:, M - 1] = np.ones(2 * M)
        matrix[:, M] = (
            sigma_T[x_i - 1] * (x - x_0) / varepsilon[x_i - 1] - mu[:]
        )

    return matrix


def psi0(x, x_i, sigma_a, q, sigma_T, mu, varepsilon, M):
    if abs(sigma_a[x_i - 1]) > 1e-8:

        psi0 = q[x_i - 1] * np.ones(2 * M) / sigma_a[x_i - 1]
    else:
        qt = q[x_i - 1]
        psi0 = (
            -1.5 * sigma_T[x_i - 1] * qt * x * x
            + 3 * varepsilon[x_i - 1] * qt * mu * x
            - 3
            * varepsilon[x_i - 1] ** 2
            * qt
            * np.power(mu, 2)
            / sigma_T[x_i - 1]
            + varepsilon[x_i - 1] ** 2
            * q[x_i - 1]
            * np.ones(2 * M)
            / sigma_T[x_i - 1]
        )
    return psi0


def solve(
    xs: Tuple[Union[int, float]],
    mus: Tuple[np.ndarray],
    psi_bc: Callable,
    coeff_fns: Tuple[Callable],
):

    xl, xr, N = xs
    mu, omega, mu_l, mu_r = mus
    M = int(mu.shape[0] / 2)

    psi_r = psi_bc(mu[0:M])
    psi_l = psi_bc(mu[M:])

    f_sigma_T, f_sigma_a, f_varepsilon = coeff_fns

    h = (xr - xl) / N
    varepsilon = np.zeros([N, 1])
    sigma_T = np.zeros([N, 1])
    sigma_a = np.zeros([N, 1])
    q = np.zeros([N, 1])

    for i in range(1, N + 1):
        varepsilon[i - 1] = (
            integrate.quad(f_varepsilon, (i - 1) * h, i * h)[0] / h
        )
        sigma_T[i - 1] = integrate.quad(f_sigma_T, (i - 1) * h, i * h)[0] / h
        sigma_a[i - 1] = integrate.quad(f_sigma_a, (i - 1) * h, i * h)[0] / h
        q[i - 1] = integrate.quad(f_q, (i - 1) * h, i * h)[0] / h

    ssl = np.zeros([2 * M, 2 * M, N])
    ssr = np.zeros([2 * M, 2 * M, N])
    psi0l = np.zeros([2 * M, N])
    psi0r = np.zeros([2 * M, N])

    for i in range(1, N + 1):
        ssl[:, :, i - 1] = sss(
            (i - 1) * h, i, varepsilon, h, sigma_T, xl, mu, M, sigma_a, omega
        )
        ssr[:, :, i - 1] = sss(
            i * h, i, varepsilon, h, sigma_T, xl, mu, M, sigma_a, omega
        )
        psi0l[:, i - 1] = psi0(
            (i - 1) * h, i, sigma_a, q, sigma_T, mu, varepsilon, M
        ).flatten()
        psi0r[:, i - 1] = psi0(
            i * h, i, sigma_a, q, sigma_T, mu, varepsilon, M
        ).flatten()

    Mt = np.zeros([2 * M, 2 * M, N])
    bt = np.zeros([2 * M, N])
    Mt[0:M, :, 0] = ssl[M : 2 * M, :, 0]
    bt[0:M, 0] = psi_l - psi0l[M : 2 * M, 0]
    Mt[M : 2 * M, :, N - 1] = ssr[0:M, :, N - 1]
    bt[M : 2 * M, N - 1] = psi_r - psi0r[0:M, N - 1]

    for i in range(1, N):
        Q, _ = linalg.qr(np.vstack((ssr[:, :, i - 1], Mt[0:M, :, i - 1])))
        Q = Q.transpose()
        Mt[0:M, :, i] = Q[2 * M :, 0 : 2 * M] @ ssl[:, :, i]
        bt[0:M, i] = (
            -Q[2 * M :, 0 : 2 * M] @ (psi0l[:, i] - psi0r[:, i - 1])
            - Q[2 * M :, 2 * M :] @ bt[0:M, i - 1]
        )

    for i in range(N, 1, -1):
        Q, _ = linalg.qr(
            np.vstack((ssl[:, :, i - 1], Mt[M : 2 * M, :, i - 1]))
        )
        Q = Q.transpose()
        Mt[M : 2 * M, :, i - 2] = Q[2 * M :, 0 : 2 * M] @ ssr[:, :, i - 2]
        bt[M : 2 * M, i - 2] = (
            -Q[2 * M :, 0 : 2 * M] @ (psi0r[:, i - 2] - psi0l[:, i - 1])
            - Q[2 * M :, 2 * M :] @ bt[M : 2 * M, i - 1]
        )

    alpha = np.zeros((2 * M, N))
    for i in range(1, N + 1):
        alpha[:, i - 1] = np.linalg.inv(Mt[:, :, i - 1]) @ bt[:, i - 1]

    res = np.zeros((2 * M, N - 1))
    for i in range(1, N):
        res[:, i - 1] = (
            ssr[:, :, i - 1] @ alpha[:, i - 1]
            + psi0r[:, i - 1]
            - ssl[:, :, i] @ alpha[:, i]
            - psi0l[:, i]
        )

    psi = np.zeros((2 * M, N + 1))
    for i in range(1, N + 1):
        psi[:, i - 1] = ssl[:, :, i - 1] @ alpha[:, i - 1] + psi0l[:, i - 1]

    psi[:, N] = ssr[:, :, N - 1] @ alpha[:, N - 1] + psi0r[:, N - 1]

    # compute correct psi_l, psi_r
    psi_l = psi_bc(mu_l)
    psi_r = psi_bc(mu_r)

    out_dict = {
        "sigma_T": sigma_T,
        "sigma_a": sigma_a,
        "psi_l": psi_l,
        "psi_r": psi_r,
        "psi": psi,
    }
    return out_dict


def save_data(dir, dirs="./data_30_100_x2", name=""):
    dir_data = dirs + f"/M={dir['M'][0]}-N={dir['N'][0]}"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if not platform.system() == "Windows":
        shutil.copy(__file__, f"{dirs}/{os.path.basename(__file__)}")
    np.savez(dir_data, **dir)


def f_varepsilon(x):
    return 1.0


def f_q(x):
    return 0.0


def f_sigma_T(b, x):

    return (
        np.tanh(11 * (6.5 / 11 - x / 2)) + np.tanh(11 * (x / 2 - 4.5 / 11))
    ) / 2 + b


def f_sigma_a(x):

    return 1 * np.ones_like(x)


def phi(x, a1=1.0, a2=0.1, k=1.0):
    # M_phi = np.zeros(M)
    # assert x.shape == np.zeros(M).shape
    m_phi = a1 * np.cos(k * x) + a2 * np.sin(k * x) + 2.0
    # M_phi = abs(0.5 - ((a+1)/2) * abs(x))

    return m_phi


def run(cfg: ConfigDict, data_path="./data/test"):

    mu, omega0 = qnwlege1(2 * cfg.num_mu, -1.0, 1.0)
    omega = omega0 / 2.0
    mu_l, omega_l = qnwlege1(cfg.num_mu, 0.0, 1.0)
    mu_r, omega_r = qnwlege1(cfg.num_mu, -1.0, 0.0)

    x = np.linspace(cfg.xmin, cfg.xmax, cfg.num_cells + 1)

    bs = np.arange(2.0, 3.0, 0.01)

    cfg.update(
        {
            "x": x,
            "mu": mu,
            "omega": omega,
            "mu_l": mu_l,
            "omega_l": omega_l,
            "mu_r": mu_r,
            "omega_r": omega_r,
            "b": bs,
        }
    )

    data = {
        "psi_l": [],
        "psi_r": [],
        "sigma_T": [],
        "sigma_a": [],
        "psi": [],
    }
    for b in tqdm(bs):
        outputs = solve(
            (cfg.xmin, cfg.xmax, cfg.num_cells),
            (mu, omega, mu_l, mu_r),
            psi_bc=phi,
            coeff_fns=(
                functools.partial(f_sigma_T, b),
                f_sigma_a,
                f_varepsilon,
            ),
        )

        for k, v in data.items():
            v.append(outputs[k].copy())

    for k in data.keys():
        data[k] = np.asarray(data[k])

    data.update(cfg)

    np.savez(data_path, **data)


if __name__ == "__main__":

    config = ml_collections.ConfigDict(
        {"xmin": 0.0, "xmax": 2.0, "num_cells": 100, "num_mu": 30}
    )
    run(config)
