import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray

XC, YC = -.4, 0
SIZE_X, SIZE_Y = 4, 4
X0, Y0 = XC - SIZE_X/2, YC - SIZE_Y/2

PX_PER_UNIT = 300
PTS_X, PTS_Y = int(SIZE_X * PX_PER_UNIT), int(SIZE_Y * PX_PER_UNIT)

BoundType = tuple[tuple[float, float], tuple[float, float]]
BOUNDS: BoundType = ((X0, Y0), (X0 + SIZE_X, Y0 + SIZE_Y))

LX = np.linspace(X0, X0 + SIZE_X, PTS_X)
LY = np.linspace(Y0, Y0 + SIZE_Y, PTS_Y)
GX, GY = np.meshgrid(LX, LY, indexing="ij")

CPLANE = GX + 1j * GY
POINTS_COUNT = np.zeros(CPLANE.shape, dtype=np.uint32)

def point2coords(pt: NDArray) -> tuple[int | None, int | None]:
    real: ArrayLike = pt.real.flatten()
    xi = (real - X0) / SIZE_X * PTS_X
    xok = (xi >= 0) & (xi < PTS_X) & (real != np.nan)
    
    imag: ArrayLike = pt.imag.flatten()
    yi = (imag - Y0) / SIZE_Y * PTS_Y
    yok = (yi >= 0) & (yi < PTS_Y) & (imag != np.nan)
    
    within_img = xok & yok
    
    return (
		xi[within_img].astype(np.uint16),
		yi[within_img].astype(np.uint16),
	)

def znp1(z: complex, c: complex) -> complex:
    return z**2 + c

def do_batch(iterations: int, nb_c: int, bounds: BoundType):
    (x0, y0), (x1, y1) = bounds
    # FIXME: first make a mandelbrot set. then choose points that are outside!
    real, imag = np.random.uniform(x0, x1, nb_c), np.random.uniform(y0, y1, nb_c)
    c_sample = real + 1j * imag
    ziter = np.zeros(c_sample.shape)
    
    # let's make a first full iteration to see which one do diverge
    for it in range(iterations):
        ziter = znp1(ziter, c_sample)
    
    # we then filter the ones that converge
    divergent = ziter.real**2 + ziter.imag**2 >= 4
    div_c_sample = c_sample[divergent]
    div_ziter = np.zeros(div_c_sample.shape)
    
    # and re-do the iteration for the ones that do diverge
    for it in range(iterations):
        div_ziter = znp1(div_ziter, div_c_sample)
        xi, yi = point2coords(div_ziter)
        POINTS_COUNT[xi, yi] += 1
    
    print("sample over")

for i in range(100):
    do_batch(100, 100_000, BOUNDS)
    
plt.matshow(POINTS_COUNT)
plt.show()
