# Module 3: Logistic Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Notebook](https://img.shields.io/badge/Jupyter-Module__3.ipynb-orange)
![Status](https://img.shields.io/badge/Solution-Complete-brightgreen)

A hands-on notebook that walks through the **sigmoid function**, the **decision boundary**, and training a **logistic regression model with scikit-learn**.

> **Tip:** Click any **▶ arrow** below to expand a section. Tick the checkboxes as you go.

---

## 📑 Table of Contents

1. [What you will learn](#-what-you-will-learn)
2. [Quick start](#-quick-start)
3. [Project structure](#-project-structure)
4. [Walkthrough](#-walkthrough)
5. [Results](#-results)
6. [Progress checklist](#-progress-checklist)
7. [FAQ](#-faq)

---

## 🎯 What you will learn

- [x] Implement the sigmoid (logistic) function
- [x] Visualize how `sigmoid(z)` maps any value into the range 0 to 1
- [x] Understand and plot a decision boundary
- [x] Train a logistic regression model with `scikit-learn`
- [x] Get predictions, accuracy, coefficients and intercept

---

## 🚀 Quick start

<details>
<summary><b>Option A: Run in Google Colab (no setup)</b></summary>

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook** and select `Module_3.ipynb`.
3. Upload `utils.py` through the Files panel on the left (it is lost when the runtime resets, so re-upload it each session).
4. Choose **Runtime → Run all**.

</details>

<details>
<summary><b>Option B: Run locally</b></summary>

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy matplotlib scikit-learn jupyter

jupyter notebook Module_3.ipynb
```

</details>

---

## 🗂 Project structure

```
.
├── Module_3.ipynb   # Notebook with the complete solution
├── utils.py         # Helper providing plot_data() (required by the notebook)
└── README.md        # This file
```

---

## 🧭 Walkthrough

<details>
<summary><b>1. Sigmoid function</b></summary>

The sigmoid maps any real number to a value between 0 and 1:

```
g(z) = 1 / (1 + e^(-z))
```

```python
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
```

It works for both a single number and a NumPy array, because `np.exp` is applied element-wise.

**Sample output**

| z   | sigmoid(z) |
|-----|------------|
| -10 | 0.000      |
| -2  | 0.119      |
| 0   | 0.500      |
| 2   | 0.881      |
| 10  | 1.000      |

</details>

<details>
<summary><b>2. Decision boundary</b></summary>

Training data: 6 examples, 2 features each.

```python
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
```

With the given parameters `b = -3`, `w0 = 1`, `w1 = 1`, the model is:

```
h(x) = g(x0 + x1 - 3)
```

The model predicts `y = 1` when `x0 + x1 - 3 >= 0`, so the boundary is the line:

```
x1 = 3 - x0
```

| Condition   | Prediction |
|-------------|------------|
| `h(x) >= 0.5` | `y = 1`  |
| `h(x) < 0.5`  | `y = 0`  |

</details>

<details>
<summary><b>3. Train with scikit-learn</b></summary>

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y.ravel())

y_pred = model.predict(X)
print(y_pred)

print(model.score(X, y.ravel()))

print(model.coef_)
print(model.intercept_)
```

| Step | Method |
|------|--------|
| Create the model | `LogisticRegression()` |
| Train | `model.fit(X, y.ravel())` |
| Predict | `model.predict(X)` |
| Accuracy | `model.score(X, y.ravel())` |
| Weights (w0, w1) | `model.coef_` |
| Bias (b) | `model.intercept_` |

</details>

---

## 📊 Results

<details open>
<summary><b>Expected output of the scikit-learn section</b></summary>

| Metric | Value |
|--------|-------|
| Predictions | `[0 0 0 1 1 1]` |
| Accuracy | `1.0` |
| Coefficients (w0, w1) | `[[0.904 0.736]]` |
| Intercept (b) | `[-2.334]` |

</details>

<details>
<summary><b>Why are the learned weights different from b = -3, w0 = 1, w1 = 1?</b></summary>

scikit-learn applies **L2 regularization** by default (`C=1.0`), which shrinks the weights. Both sets of parameters classify all six training points correctly, so accuracy is 100% either way.

</details>

---

## ✅ Progress checklist

Copy this list into your own notes and tick items as you finish them.

- [ ] Placed `utils.py` next to the notebook
- [ ] Imported `numpy`, `matplotlib` and `utils.plot_data`
- [ ] Completed the `sigmoid(z)` function
- [ ] Plotted `z` vs `sigmoid(z)`
- [ ] Plotted the decision boundary `x1 = 3 - x0`
- [ ] Trained `LogisticRegression` on `X`, `y`
- [ ] Printed predictions, score, coefficients and intercept

---

## ❓ FAQ

<details>
<summary><b>I get <code>ModuleNotFoundError: No module named 'utils'</code></b></summary>

The notebook imports `plot_data` from `utils.py`, which must sit in the same folder as the notebook (in Colab, upload it through the Files panel). Use the `utils.py` file included in this repo:

```python
import numpy as np


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc="best"):
    y = np.asarray(y).ravel()
    pos = y == 1
    neg = y == 0

    ax.scatter(X[pos, 0], X[pos, 1], marker="x", s=s, c="red", label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker="o", s=s, facecolors="none", edgecolors="blue", linewidths=2, label=neg_label)
    ax.legend(loc=loc)
```

It draws `y=1` as red **x** markers and `y=0` as blue hollow circles.

</details>

<details>
<summary><b>Why <code>y.ravel()</code> in <code>fit</code> and <code>score</code>?</b></summary>

`y` is stored as a column vector with shape `(6, 1)`. scikit-learn expects shape `(6,)`, and `ravel()` flattens it. Without it you get a `DataConversionWarning`.

</details>

<details>
<summary><b>The sigmoid plot does not appear</b></summary>

Add `plt.show()` at the end of the plotting cell. Some environments display plots automatically, others do not.

</details>

---

## 📚 References

- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [NumPy: `exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)
