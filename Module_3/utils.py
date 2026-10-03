import numpy as np


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc="best"):
    y = np.asarray(y).ravel()
    pos = y == 1
    neg = y == 0

    ax.scatter(X[pos, 0], X[pos, 1], marker="x", s=s, c="red", label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker="o", s=s, facecolors="none", edgecolors="blue", linewidths=2, label=neg_label)
    ax.legend(loc=loc)
