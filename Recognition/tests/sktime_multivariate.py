from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.datasets import load_basic_motions
import matplotlib.pyplot as plt
import numpy as np


X, y = load_basic_motions(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
np.unique(y_train)

fig, axs = plt.subplots(len(X_test.to_numpy()[0]))
for i,v in enumerate(X_test.to_numpy()[0]):
    axs[i].plot(v)

plt.show()

steps = [
    ("concatenate", ColumnConcatenator()),
    ("classify", TimeSeriesForestClassifier(n_estimators=100)),
]

clf = Pipeline(steps)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
