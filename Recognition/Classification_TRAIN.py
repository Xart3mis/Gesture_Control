import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from pyts.preprocessing import MinMaxScaler
from pyts.multivariate.transformation import MultivariateTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.kernel_based import Arsenal
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["right"].set_visible(False)


path = "/home/xart3misx/Documents/Gesture_Control/data"

dirs = []

x = []
y = []

labels = {}

for i, v in enumerate(list(next(os.walk(path))[1])):
    labels[i] = v


for i in [
    path + "/" + list(next(os.walk(path))[1])[i]
    for i in range(len(list(next(os.walk(path))[1])))
]:
    for j in os.walk(i):
        for k in j[2]:
            data = pd.read_csv(i + "/" + k)
            _h, _r, _p, _ax, _ay, _az, _mx, _my, _mz = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
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
        x[i][i1] = interp.interp1d(np.linspace(0, 99, num=len(v1)), v1, kind="cubic")(
            np.linspace(0, 99)
        )


def smooth(scalars, weight=0.75):  # Weight between 0 and 1
    return [
        scalars[i] * weight + (1 - weight) * scalars[i + 1]
        for i in range(len(scalars))
        if i < len(scalars) - 1
    ]


x = np.asarray(x)
x = MultivariateTransformer(
    MinMaxScaler(sample_range=(-1, 1)), flatten=False
).fit_transform(x)

x_concat = []
for i in x:
    x_concat.append(
        [
            smooth(np.concatenate((i[0], i[1], i[2]), axis=None)),
            smooth(np.concatenate((i[3], i[4], i[5]), axis=None)),
            smooth(np.concatenate((i[6], i[7], i[8]), axis=None)),
        ]
    )

x = np.array(x_concat)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

def fit_models(_x, _y):
    estimators = [
    (
        "TSFC",
        TimeSeriesForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0),
        0,
    ),
    ("WEASEL", WEASEL(random_state=0), 0),
    ("cBOSS", ContractableBOSS(random_state=0), 0),
    ("ShapeletTransform", ShapeletTransformClassifier(), 0),
    ("Arsenal", Arsenal(), 0)
]

    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", ColumnEnsembleClassifier(estimators=estimators)),
    ]

    print("Fitting Classifier Stack.")
    print(
        "Classifiers:",
        "".join(
            [
                (v[0] + ", ") if i < len(estimators) - 1 else (v[0])
                for i, v in enumerate(estimators)
            ]
        ),
    )
    clf = Pipeline(steps)
    clf.fit(_x, _y)
    print("Done.")

    pickle.dump(
        [labels, clf],
        open(
            "/home/xart3misx/Documents/Gesture_Control/Models/Classsifier.pickle", "wb"
        ),
    )

    return clf


def plot_and_show():
    idx = 0

    fig = plt.figure()
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")
    axs[0, 0].plot(x_concat[idx][0])
    axs[0, 1].plot(x_concat[idx][1])
    axs[0, 2].plot(x_concat[idx][2])
    axs[1, 0].plot(x_concat[idx + 1][0])
    axs[1, 1].plot(x_concat[idx + 1][1])
    axs[1, 2].plot(x_concat[idx + 1][2])
    axs[2, 0].plot(x_concat[idx + 2][0])
    axs[2, 1].plot(x_concat[idx + 2][1])
    axs[2, 2].plot(x_concat[idx + 2][2])
    for ax in axs.flat:
        ax.label_outer()


    plt.show()

if __name__ == "__main__":
    clf = fit_models(x, y)
    # print(clf.score(X_test, y_test))