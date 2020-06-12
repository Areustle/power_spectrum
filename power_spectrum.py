import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import scipy.stats as stats
import pylab as pl
import time

image = mpimg.imread("big_milky_way_blue.png")

# print(image.shape)
# pl.imshow(image)
# pl.show()

image = cp.asarray(image)

tstart = time.time()
fourier_image = cp.fft.fftn(image)
cp.cuda.Stream.null.synchronize()
tstop = time.time()

# print(s.message)
print(f"FFT runtime: {(tstop - tstart)} seconds. ")

# fourier_image = cp.asnumpy(fourier_image)

fourier_amplitudes = cp.abs(fourier_image)**2
fourier_amplitudes = cp.asnumpy(fourier_amplitudes).flatten()

kfreq = np.fft.fftfreq(image.shape[0])*image.shape[0]
kfreq2D = np.meshgrid(kfreq, kfreq)
knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()

kbins = np.arange(0.5, 501., 1.)
kvals = 0.5*(kbins[1:] + kbins[:-1])

Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic="mean", bins=kbins)

Abins *= 4.*np.pi/3.*(kbins[1:]**3 - kbins[:-1]**3)

plt.loglog(kvals, Abins)
plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.title("Milky Way Galaxy Core Power Spectrum (Blue)")
plt.tight_layout()
plt.savefig("milky_way_blue_power_spectrum.png", dpi=300, bbox_inches="tight")
