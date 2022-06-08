from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import preprocessing
from dtaidistance import dtw_ndim_visualisation
from dtaidistance import dtw_ndim
from dtaidistance import dtw
from more_itertools import sample
from scipy import stats
import pandas as pd
import numpy as np


s1 = []
for h, r, p in zip(stats.zscore(heading1), stats.zscore(roll1), stats.zscore(pitch1)):
    s1.append([h, r, p])


s2 = []
for h, r, p in zip(stats.zscore(heading2), stats.zscore(roll2), stats.zscore(pitch2)):
    s2.append([h, r, p])


s1 = np.array(s1, dtype=np.double)
s2 = np.array(s2, dtype=np.double)
s1 = preprocessing.differencing(s1)
s2 = preprocessing.differencing(s2)

d = dtw_ndim.distance(s1, s2, window=2, use_pruning=True)

print(d)
# dtw_ndim_visualisation.plot_warping(s1, s2, dtw_ndim.warping_path(s1, s2, window=2), filename="warp.png")
