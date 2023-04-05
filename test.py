# %%
from Datasets.utils import make_intrinsics_layer
import numpy as np
# %%

h, w = (448, 640)
fx = 707.0912
fy = 707.0912
cx = 601.8873
cy = 183.1104
intrinsicLayer = make_intrinsics_layer(w, h, fx, fy, cx, cy)

# %%
A = np.meshgrid(range(640), range(448))
# %%
