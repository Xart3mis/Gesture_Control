import matplotlib.pyplot as plt
import numpy as np

from tslearn.metrics import dtw_path, ctw_path


def plot_trajectory(ts, ax, color_code=None, alpha=1.0):
    if color_code is not None:
        colors = [color_code] * len(ts)
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
    for i in range(len(ts) - 1):
        ax.plot(ts[i : i + 2, 0], ts[i : i + 2, 1], marker="o", c=colors[i], alpha=alpha)


def get_rot2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def make_one_folium(sz, a=1.0, noise=0.1, resample_fun=None):
    theta = np.linspace(0, 1, sz)
    if resample_fun is not None:
        theta = resample_fun(theta)
    theta -= 0.5
    theta *= 0.9 * np.pi
    theta = theta.reshape((-1, 1))
    r = a / 2 * (4 * np.cos(theta) - 1.0 / np.cos(theta))
    x = r * np.cos(theta) + np.random.rand(sz, 1) * noise
    y = r * np.sin(theta) + np.random.rand(sz, 1) * noise
    return np.array(np.hstack((x, y)))


trajectory = make_one_folium(sz=30).dot(get_rot2d(np.pi + np.pi / 3))
rotated_trajectory = trajectory.dot(get_rot2d(np.pi / 4)) + np.array([0.0, 3.0])

path_dtw, _ = dtw_path(trajectory, rotated_trajectory)

path_ctw, cca, _ = ctw_path(trajectory, rotated_trajectory, max_iter=100, n_components=2)

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
for (i, j) in path_dtw:
    ax.plot(
        [trajectory[i, 0], rotated_trajectory[j, 0]],
        [trajectory[i, 1], rotated_trajectory[j, 1]],
        color="g" if i == j else "r",
        alpha=0.5,
    )
plot_trajectory(trajectory, ax)
plot_trajectory(rotated_trajectory, ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("DTW")

ax = plt.subplot(1, 2, 2)
for (i, j) in path_ctw:
    ax.plot(
        [trajectory[i, 0], rotated_trajectory[j, 0]],
        [trajectory[i, 1], rotated_trajectory[j, 1]],
        color="g" if i == j else "r",
        alpha=0.5,
    )
plot_trajectory(trajectory, ax)
plot_trajectory(rotated_trajectory, ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("CTW")

plt.tight_layout()
plt.show()
