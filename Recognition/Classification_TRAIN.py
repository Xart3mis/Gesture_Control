import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from pyts.classification import TSBF
from pyts.classification import BOSSVS
from pyts.classification import SAXVSM
from pyts.preprocessing import MinMaxScaler
from pyts.classification import TimeSeriesForest
from pyts.classification import LearningShapelets
from pyts.multivariate.classification import MultivariateClassifier
from pyts.multivariate.transformation import MultivariateTransformer

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.style.use("dark_background")

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
            x.append([_h, _r, _p, _ax, _ay, _az, _mx, _my, _mz])
            for l in labels.items():
                if l[1] in k:
                    y.append(l[0])

# clf = MultivariateClassifier(BOSSVS())

for i, v in enumerate(x):
    for i1, v1 in enumerate(v):
        x[i][i1] = interp.interp1d(np.linspace(0, 99, num=len(v1)), v1, kind="cubic")(np.linspace(0, 99))


x = np.asarray(x)
x = MultivariateTransformer(MinMaxScaler(sample_range=(-1, 1)), flatten=False).fit_transform(x)


def get_testx():
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

    test_x.append([_h, _r, _p, _ax, _ay, _az, _mx, _my, _mz])

    for i, v in enumerate(test_x):
        for i1, v1 in enumerate(v):
            test_x[i][i1] = interp.interp1d(np.linspace(0, 99, num=len(v1)), v1, kind="cubic")(np.linspace(0, 99))

    test_x = np.asarray(test_x)
    test_x = MultivariateTransformer(MinMaxScaler(sample_range=(-1, 1)), flatten=False).fit_transform(test_x)

    return test_x


if __name__ == "__main__":
    # print("Training Model...")
    # clf.fit(x, y)
    # print("Done Training.")

    # print(clf.score(x, y))
    test_x = get_testx()

    # print(labels[clf.predict(test_x)[0]])

    name = "Accent"
    ax2.set_prop_cycle(color=plt.cm.get_cmap(name).colors)
    ax3.set_prop_cycle(color=plt.cm.get_cmap(name).colors)

    models = [
        MultivariateClassifier(TimeSeriesForest()),
        MultivariateClassifier(BOSSVS()),
        MultivariateClassifier(SAXVSM()),
        MultivariateClassifier(TSBF()),
    ]

    print("Training Classifier Stack")
    preds = []
    for m in models:
        m.fit(x, y)
        preds.append(m.predict(test_x)[0])
    print("Done")

    print(labels[max(set(preds), key = preds.count)])

    for i in x[9]:
        ax1.plot(i, alpha=0.7)

    for i in x[0]:
        ax2.plot(i, alpha=0.7)

    for i in x[-1]:
        ax3.plot(i, alpha=0.7)


    pickle.dump([labels, models], open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "wb"))
    
    plt.tight_layout()
    plt.show()
