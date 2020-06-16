import numpy as np
import scipy as sp
import scipy.linalg as spl
import cupy as cp
from timeit import timeit
import matplotlib.pyplot as plt


def test_perf(f, xp, input_xp, max_ms=1000, max_exp=14):
    n, t, except_flag = [], [], False
    for e in range(2, max_exp):
        if not except_flag:
            if not t or t[-1] < max_ms:
                N = 2**e
                A = input_xp.random.uniform(0, 255, (N, N))
                b = input_xp.random.uniform(0, 255, (N))
                try:
                    tt = timeit('f(xp.asarray(A), xp.asarray(b))',
                                globals=locals(), number=10)*100
                except:
                    print("Exception raised")
                    except_flag = True

                if not except_flag:
                    n.append(N)
                    t.append(tt)

    return (n, t)


def cpsolvesync(A, b):
    r = cp.linalg.solve(A, b)
    cp.cuda.Stream.null.synchronize()
    return r


fig, ax = plt.subplots(2)

fig.suptitle("Linear Solver Execution time")

line_numpy = test_perf(np.linalg.solve, np, np)
line_scipy = test_perf(spl.solve, sp, sp)
line_cupy = test_perf(cpsolvesync, cp, cp)

ax[0].plot(*line_numpy, label="CPU Numpy", color='navy')
ax[0].plot(*line_scipy, label="CPU Scipy", color='blue')
ax[0].plot(*line_cupy, label="GPU Cupy", color='lime')
ax[0].set_ylim(0, 2000)
ax[0].set_ylabel("Execution Time (ms)")
ax[0].set_xlabel("N")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(*line_numpy, label="CPU Numpy", color='navy')
ax[1].plot(*line_scipy, label="CPU Scipy", color='blue')
ax[1].plot(*line_cupy, label="GPU Cupy", color='lime')
ax[1].set_ylabel("Log Execution Time (log ms)")
ax[1].set_xlabel("N")
ax[1].legend()
ax[1].set_yscale("log")
ax[1].grid(True)

plt.show()
