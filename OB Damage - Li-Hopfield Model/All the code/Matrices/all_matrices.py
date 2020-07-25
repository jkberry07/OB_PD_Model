# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:46:57 2020

@author: wmmjk
"""

import numpy as np

H0_10_1D=np.load("H0_original.npy")
W0_10_1D=np.load("W0_original.npy")
#print(W0_10_1D)
#H0 and W0, respectively, for the 1D 20 unit network
#H0= [[0.3 0.9 0.  0.  0.  0.  0.  0.  0.  0.7]
#     [0.9 0.4 1.  0.  0.  0.  0.  0.  0.  0. ]
#     [0.  0.8 0.3 0.8 0.  0.  0.  0.  0.  0. ]
#     [0.  0.  0.7 0.5 0.9 0.  0.  0.  0.  0. ]
#     [0.  0.  0.  0.8 0.3 0.8 0.  0.  0.  0. ]
#     [0.  0.  0.  0.  0.7 0.3 0.9 0.  0.  0. ]
#     [0.  0.  0.  0.  0.  0.7 0.4 0.9 0.  0. ]
#     [0.  0.  0.  0.  0.  0.  0.5 0.5 0.7 0. ]
#     [0.  0.  0.  0.  0.  0.  0.  0.9 0.3 0.9]
#     [0.9 0.  0.  0.  0.  0.  0.  0.  0.8 0.3]]
#
#W0= [[0.3 0.7 0.  0.  0.  0.  0.  0.  0.5 0.3]
#     [0.3 0.2 0.5 0.  0.  0.  0.  0.  0.  0.7]
#     [0.  0.1 0.3 0.5 0.  0.  0.  0.  0.  0. ]
#     [0.  0.5 0.2 0.2 0.5 0.  0.  0.  0.  0. ]
#     [0.5 0.  0.  0.5 0.1 0.9 0.  0.  0.  0. ]
#     [0.  0.  0.  0.  0.3 0.3 0.5 0.4 0.  0. ]
#     [0.  0.  0.  0.6 0.  0.2 0.3 0.5 0.  0. ]
#     [0.  0.  0.  0.  0.  0.  0.5 0.3 0.5 0. ]
#     [0.  0.  0.  0.  0.  0.2 0.  0.2 0.3 0.7]
#     [0.7 0.  0.  0.  0.  0.  0.  0.2 0.3 0.5]]




H0_10_2D=np.load("H0_10_2D_65Hz.npy")
W0_10_2D=np.load("W0_10_2D_65Hz.npy")
#print(H0_10_2D.round(3))
#print(W0_10_2D.round(3))
#H0 and W0 for the 2D 20 unit network
#H0= [[0.188 0.736 0.    0.    0.69  0.346 0.    0.    0.    0.72 ]
#     [0.907 0.332 1.069 0.    0.    0.895 0.833 0.    0.    0.   ]
#     [0.    0.933 0.195 0.857 0.    0.    0.169 0.327 0.    0.   ]
#     [0.    0.    0.851 0.385 0.721 0.    0.    0.15  0.857 0.   ]
#     [0.    0.    0.    0.892 0.251 0.749 0.    0.    0.641 0.256]
#     [0.642 0.    0.    0.    0.753 0.295 0.834 0.    0.    0.321]
#     [0.618 0.426 0.    0.    0.    0.713 0.438 0.893 0.    0.   ]
#     [0.    1.045 0.637 0.    0.    0.    0.42  0.514 0.822 0.   ]
#     [0.    0.    0.412 0.611 0.    0.    0.    1.012 0.506 1.055]
#     [1.097 0.    0.    0.421 0.709 0.    0.    0.    0.665 0.146]]
#
#W0= [[0.24  0.505 0.    0.    0.58  0.682 0.    0.    0.    0.215]
#     [0.284 0.1   0.612 0.    0.    0.804 0.165 0.    0.    0.   ]
#     [0.    0.25  0.29  0.495 0.    0.    0.433 0.115 0.    0.   ]
#     [0.    0.    0.27  0.197 0.456 0.    0.    0.518 0.238 0.   ]
#     [0.    0.    0.    0.454 0.032 0.743 0.    0.    0.181 0.91 ]
#     [0.193 0.    0.    0.    0.172 0.408 0.45  0.    0.    0.431]
#     [0.668 0.647 0.    0.    0.    0.069 0.309 0.475 0.    0.   ]
#     [0.    0.606 0.385 0.    0.    0.    0.595 0.299 0.467 0.   ]
#     [0.    0.    0.425 0.308 0.    0.    0.    0.302 0.482 0.61 ]
#     [0.873 0.    0.    0.911 0.461 0.    0.    0.    0.269 0.312]]



H0_20_1D=np.load("H0_20_52Hz.npy")
W0_20_1D=np.load("W0_20_52Hz.npy")
#print(H0_20_1D.round(3))
#print(W0_20_1D.round(3))
#H0 and W0 for the 1D 40 unit network
#H0= [[0.322 0.588 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.521]
#     [0.536 0.445 0.871 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.737 0.178 0.979 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.431 0.57  0.82  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.715 0.27  0.574 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.582 0.352 0.454 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    1.046 0.306 0.666 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.398 0.199 1.102 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.833 0.417 0.597 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.731 0.626 0.952 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.519 0.642 1.081 0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.662 0.716 0.92  0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.095 0.276 1.029 0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.719 0.646 0.732 0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.548 0.112 0.877 0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.728 0.524 0.98  0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.554 0.275 1.182 0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.723 0.249 0.5   0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.952 0.355 0.919]
#     [0.861 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.149 0.071]]
#
#W0= [[0.054 0.61  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.177]
#     [0.015 0.119 0.524 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.119 0.551 0.506 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.153 0.503 0.518 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.487 0.167 1.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.391 0.608 0.694 0.    0.    0.    0.    0     0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.149 0.239 0.717 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.332 0.617 0.135 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.372 0.167 0.651 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.243 0.224 0.86  0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.407 0.289 0.187 0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.231 0.498 0.444 0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.214 0.259 0.623 0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.332 0.066 0.313 0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.284 0.1   1.163 0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.161 0.628 0.542 0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.1   0.394 0.366 0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.271 0.134 0.742 0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.1   0.566 0.883]
#     [1.054 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.148 0.1  ]]

H0_20_2D=np.load("H0_20_2D_50Hz.npy")
W0_20_2D=np.load("W0_20_2D_50Hz.npy")
#print(H0_20_2D.round(3))
#print(W0_20_2D.round(3))
#H0= [[0.179 0.613 0.    0.562 0.615 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.519 0.    0.    0.   ]
#     [0.93  0.157 0.396 0.    0.    0.981 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.055 0.    0.   ]
#     [0.    0.129 0.522 0.979 0.    0.    1.247 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.717 0.   ]
#     [0.827 0.    0.221 0.541 0.    0.    0.    1.021 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.367]
#     [0.841 0.    0.    0.    0.014 0.428 0.    0.993 0.975 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.1   0.    0.    0.51  0.205 0.656 0.    0.    0.941 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.845 0.    0.    0.771 0.67  0.685 0.    0.    0.197 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.602 0.74  0.    0.97  0.281 0.    0.    0.    0.545 0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.561 0.    0.    0.    0.181 0.683 0.    0.345 0.492 0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.355 0.    0.    0.761 0.469 0.304 0.    0.    0.133 0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.658 0.    0.    0.808 0.117 0.554 0.    0.    1.177 0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.373 0.351 0.    0.239 0.191 0.    0.    0.    0.809 0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.929 0.    0.    0.    0.1   0.95  0.    0.747 0.953 0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.622 0.    0.    0.053 0.467 0.431 0.    0.    0.705 0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.558 0.    0.    0.56  0.376 0.255 0.    0.    1.173 0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.458 0.815 0.    0.37  0.175 0.    0.    0.    0.403]
#     [0.992 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.038 0.    0.    0.    0.6   1.023 0.    1.   ]
#     [0.    0.846 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.391 0.    0.    1.236 0.522 0.703 0.   ]
#     [0.    0.    0.761 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.651 0.    0.    0.294 0.286 0.439]
#     [0.    0.    0.    0.695 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.22  0.795 0.    0.979 0.508]]
#    
#W0= [[0.202 0.34  0.    0.546 0.351 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.595 0.    0.    0.   ]
#     [0.927 0.141 0.434 0.    0.    0.549 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.197 0.    0.   ]
#     [0.    0.487 0.269 0.353 0.    0.    0.103 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.069 0.   ]
#     [1.315 0.    0.2   0.311 0.    0.    0.    0.159 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.639]
#     [0.148 0.    0.    0.    0.322 1.22  0.    0.471 0.789 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.436 0.    0.    0.569 0.2   0.355 0.    0.    0.865 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.556 0.    0.    0.887 0.492 0.314 0.    0.    0.356 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.369 0.57  0.    0.872 0.151 0.    0.    0.    0.824 0.    0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.157 0.    0.    0.    0.057 0.256 0.    0.444 0.46  0.    0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.318 0.    0.    0.167 0.126 0.349 0.    0.    0.575 0.    0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.197 0.    0.    0.322 0.522 0.77  0.    0.    0.074 0.    0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.562 0.038 0.    0.38  0.16  0.    0.    0.    0.12  0.    0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    1.051 0.    0.    0.    0.33  0.49  0.    0.297 0.335 0.    0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.089 0.    0.    0.445 0.028 1.07  0.    0.    0.282 0.    0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.648 0.    0.    0.533 0.195 0.637 0.    0.    0.3   0.   ]
#     [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.1   0.112 0.    0.543 0.387 0.    0.    0.    0.624]
#     [0.092 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.196 0.    0.    0.    0.002 0.31  0.    0.518]
#     [0.    0.778 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.142 0.    0.    0.964 0.342 0.272 0.   ]
#     [0.    0.    0.782 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.518 0.    0.    0.822 0.16  0.554]
#     [0.    0.    0.    0.195 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.267 0.589 0.    0.094 0.351]]



#The weight matrices for the 100 unit networks are too large to print
H0_50_1D=np.load("H0_50_53Hz.npy")
W0_50_1D=np.load("W0_50_53Hz.npy")

H0_50_2D=np.load("H0_50_2D_60Hz.npy")
W0_50_2D=np.load("W0_50_2D_60Hz.npy")
