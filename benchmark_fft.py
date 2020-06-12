import numpy as np
import scipy as sp
import cupy as cp
from timeit import timeit
import matplotlib.pyplot as plt


def test_perf(f, xp, input_xp, max_ms=1000, max_exp=14):
    n, t, except_flag = [], [], False
    for e in range(2, max_exp):
        if not except_flag:
            if not t or t[-1] < max_ms:
                N = 2**e
                x = input_xp.random.uniform(0, 255, (N, N))
                try:
                    tt = timeit('f(xp.asarray(x))', globals=locals(),
                                number=10)*100
                except:
                    print("Exception raised")
                    except_flag = True

                if not except_flag:
                    n.append(N)
                    t.append(tt)

    return (n, t)


def cpfftsync(x):
    r = cp.fft.fftn(x)
    cp.cuda.Stream.null.synchronize()
    return r


fig, ax = plt.subplots(2)

fig.suptitle("FFT Execution time")

line_numpy = test_perf(np.fft.fftn, np, np)
line_scipy = test_perf(sp.fft.fftn, sp, sp)
line_cupy = test_perf(cpfftsync, cp, cp)

ax[0].plot(*line_numpy, label="CPU Numpy", color='navy')
ax[0].plot(*line_scipy, label="CPU Scipy", color='blue')
ax[0].plot(*line_cupy, label="GPU Cupy", color='lime')
ax[0].set_ylim(0, 1000)
ax[0].set_ylabel("Execution Time (ms)")
ax[0].set_xlabel("2D Image Height")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(*line_numpy, label="CPU Numpy", color='navy')
ax[1].plot(*line_scipy, label="CPU Scipy", color='blue')
ax[1].plot(*line_cupy, label="GPU Cupy", color='lime')
ax[1].set_ylabel("Log Execution Time (log ms)")
ax[1].set_xlabel("2D Image Height")
ax[1].legend()
ax[1].set_yscale("log")
ax[1].grid(True)

plt.show()
