import os
import json
import pickle
import numpy as np
from rdp import rdp
import pandas as pd
from scipy import stats
from tslearn import utils
from tslearn import neighbors
import matplotlib.pyplot as plt

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
            __x = []
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
                __x.append([h, r, p, ax, ay, az, mx, my, mz])
            # y.append()
            x.append(__x)
            for l in labels.items():
                if l[1] in k:
                    y.append(l[0])

# y = np.array(y)
# print(y.shape)
# for i, v in enumerate(x):
#     x[i] = rdp(v, epsilon=0.4)
time_series = utils.to_time_series_dataset(x)
clf = neighbors.KNeighborsTimeSeriesClassifier(metric="sax", n_neighbors=5)
print(len(y))
print(len(x))


def plot_dataset():
    for i in time_series:
        plt.plot(np.linspace(0, 100000, num=len(i)), i, marker="o")
    plt.show()


if __name__ == "__main__":
    clf.fit(time_series, y)
    data = pd.read_csv("/home/lethargic/Documents/PicoMPU9250/predict_test/test.csv")
    testx = []
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
        testx.append([h, r, p, ax, ay, az, mx, my, mz])

    time_series_testx = utils.to_time_series([testx])
    print(labels[clf.predict(time_series_testx)[0]])
    pickle.dump([clf, labels], open("/home/lethargic/Documents/PicoMPU9250/Models/Classsifier.pickle", "wb"))
    plot_dataset()
