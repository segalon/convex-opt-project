# Convex Optimization Project

import numpy as np


def solve(S, k):
    n = S.shape[0]

    eps = 1e-12
    S += eps * np.eye(n)

    if n == k:
        return np.linalg.solve(S, np.eye(n))

    R = np.zeros((n, n))
    for j in range(n):
        d = max(j - k, 0)
        if j == 0:
            coeffs = None
            n = 0
        else:
            coeffs = np.array(S[d: j, d: j+1])
            n = coeffs.shape[1]

        last_eq = np.array(S[j, d: j + 1])

        for i in range(0, n - 1):
            if i == n - 2:
                pass
            else:
                coeffs[i + 1: n, i + 1:n] -= coeffs[i + 1:n, i][:, None] * coeffs[i, i + 1: n] / coeffs[i, i]
            last_eq[i + 1: n] -= last_eq[i] * coeffs[i, i + 1: n] / coeffs[i, i]

        sub_ans = np.array([1 / np.sqrt(last_eq[n - 1])])
        for i in range(n - 2, -1, -1):
            a_00 = np.array([np.sum((-1 * sub_ans) * coeffs[i, i + 1: i + 1 + len(sub_ans) + 1] / coeffs[i, i])])
            sub_ans = np.concatenate((a_00, sub_ans))

        R[d: j + 1, j] = sub_ans

    return R @ R.T
