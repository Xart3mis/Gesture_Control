import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyts.classification import BOSSVS
from pyts.multivariate.classification import MultivariateClassifier

path = "/home/lethargic/Documents/PicoMPU9250/data"

dirs = []

x = []
y = []

labels = {}

for i, v in enumerate(list(next(os.walk(path))[1])):
    labels[i] = v


for i in [path + "/" + list(next(os.walk(path))[1])[i] for i in range(len(list(next(os.walk(path))[1])))]:
    for j in os.walk(i):
        for k in j[2]:
            data = pd.read_csv(i + "/" + k)
            _h, _r, _p, _ax, _ay, _az, _mx, _my, _mz = [], [], [], [], [], [], [], [], []
            for h, r, p, ax, ay, az, mx, my, mz in zip(
                data.heading.tolist(),
                data.roll.tolist(),
                data.pitch.tolist(),
                data.ax.tolist(),
                data.ay.tolist(),
                data.az.tolist(),
                data.mx.tolist(),
                data.my.tolist(),
                data.mz.tolist(),
            ):
                _h.append(h)
                _r.append(r)
                _p.append(p)
                _ax.append(ax)
                _ay.append(ay)
                _az.append(az)
                _mx.append(mx)
                _my.append(my)
                _mz.append(mz)

            # y.append()
            x.append([_h, _r, _p, _ax, _ay, _az])
            for l in labels.items():
                if l[1] in k:
                    y.append(l[0])

clf = MultivariateClassifier(BOSSVS())

Tnew = np.arange(0, len(max(x, key=lambda s: len(s[0]))[0]))

for i, v in enumerate(x):
    Told = np.arange(0, len(v[0]))

    F = interp1d(Told, v, fill_value="extrapolate")
    x[i] = F(Tnew)

x = np.asarray(x)

plt.show()
clf.fit(x, y)

if __name__ == "__main__":
    print(clf.score(x, y))

    data = pd.read_csv("/home/lethargic/Documents/PicoMPU9250/predict_test/test.csv")
    test_x = []
    _h, _r, _p, _ax, _ay, _az, _mx, _my, _mz = [], [], [], [], [], [], [], [], []

    for h, r, p, ax, ay, az, mx, my, mz in zip(
        data.heading.tolist(),
        data.roll.tolist(),
        data.pitch.tolist(),
        data.ax.tolist(),
        data.ay.tolist(),
        data.az.tolist(),
        data.mx.tolist(),
        data.my.tolist(),
        data.mz.tolist(),
    ):
        _h.append(h)
        _r.append(r)
        _p.append(p)
        _ax.append(ax)
        _ay.append(ay)
        _az.append(az)
        _mx.append(mx)
        _my.append(my)
        _mz.append(mz)
    test_x.append([_h, _r, _p, _ax, _ay, _az])

    for i, v in enumerate(test_x):
        Told = np.arange(0, len(v[0]))
        F = interp1d(Told, v, fill_value="extrapolate")
        test_x[i] = F(Tnew)

    test_x = np.asarray(test_x)
    print(test_x.shape)

    print(labels[clf.predict(test_x)[0]])
    pickle.dump([Tnew, labels, clf], open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "wb"))
