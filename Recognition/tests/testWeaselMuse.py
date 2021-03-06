import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from pyts.classification import BOSSVS
from pyts.preprocessing import MinMaxScaler
from pyts.multivariate.image import JointRecurrencePlot
from pyts.multivariate.transformation import WEASELMUSE
from pyts.multivariate.classification import MultivariateClassifier
from pyts.multivariate.transformation import MultivariateTransformer

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

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

clf = MultivariateClassifier(BOSSVS())

Tnew = np.arange(0, len(max(x, key=lambda s: len(s[0]))[0]))


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
    # test_x = get_testx()
    # print(test_x.shape)

    # print(labels[clf.predict(test_x)[0]])
    # pickle.dump([Tnew, labels, clf], open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "wb"))

    # # Recurrence plot transformation
    # jrp = JointRecurrencePlot(threshold="point", percentage=50)
    # X_jrp = jrp.fit_transform(x)

    # ax1.imshow(X_jrp[0], cmap="binary", origin="lower")
    # plt.tight_layout()

    # for i in x[0]:
    #     ax2.plot(i)

    # for i in x[-1]:
    #     ax3.plot(i)

    # plt.show()

    transformer = WEASELMUSE()
    X_weasel = transformer.fit_transform(x, y)

    for i in X_weasel:
        print(i)
    plt.show()
