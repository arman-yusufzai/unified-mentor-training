# Module 1: Vectors using NumPy

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arman-yusufzai/unified-mentor-training/blob/main/Module_1/Module_1.ipynb)

> Click the badge above to run the notebook in your browser. No installation needed.

**Environment:** Python 3.13.9, NumPy

---

## 📋 Progress Checklist

- [x] Read the NumPy documentation
- [x] Vector operations (add, subtract, scalar, element-wise, dot product)
- [x] Matrix inverse check (`v1 @ inv(v1)` = identity)
- [x] Universal function #1: Trigonometric (`np.sin`)
- [x] Universal function #2: Arithmetic (`np.add`)
- [x] Universal function #3: Comparison (`np.greater`, `np.equal`, `np.not_equal`)

---

## 📖 Step 1: Documentation

👉 [NumPy Quickstart: The Basics](https://numpy.org/doc/stable/user/quickstart.html#the-basics)

<details>
<summary><b>💡 Key ideas (click to expand)</b></summary>

- An `ndarray` is NumPy's fixed-type, N-dimensional array
- Key attributes: `ndim`, `shape`, `size`, `dtype`
- A **universal function (ufunc)** works on every element of an array at once, with no Python loop

</details>

---

## 🧪 Step 2: Vector Operations

```python
import numpy as np
from numpy.linalg import inv

n1 = np.array([[22, 33, 44]])
v1 = np.array([[2, 2, 2]])
v2 = np.array([[2, 3, 4]])
a  = np.array([11, 12, 13])
```

| Operation | Code | Output |
|---|---|---|
| Addition | `n1 + v1` | `[[24 35 46]]` |
| Subtraction | `n1 - v2` | `[[20 30 40]]` |
| Scalar multiplication | `n1 * 25` | `[[ 550  825 1100]]` |
| Element-wise multiplication | `np.multiply(n1, a)` | `[[242 396 572]]` |
| Dot product with transpose | `np.dot(n1, n1.T)` | `[[3509]]` |

<details>
<summary><b>🔍 Outer-style multiplication (3×1 by 1×4)</b></summary>

```python
re = n1.reshape(3, 1)
b = np.full((1, 4), 3)
np.multiply(re, b)
```

```
[[ 66  66  66  66]
 [ 99  99  99  99]
 [132 132 132 132]]
```

`np.dot` can also be written with the `@` operator.

</details>

<details>
<summary><b>🔍 Matrix times its inverse</b></summary>

```python
v1 = np.array([[2, 3, 5], [7, 11, 13], [17, 19, 23]])   # 3x3 prime matrix
v2 = inv(v1)
np.round(v1 @ v2)
```

```
[[ 1.  0. -0.]
 [ 0.  1.  0.]
 [ 0. -0.  1.]]
```

The result is the identity matrix. The `-0.` entries are harmless rounding noise.

</details>

---

## 🎯 Step 3: TASK, Universal Functions on `v1`

> From here on, `v1` is the 3×3 prime matrix above, and `q = np.array([7, 8, 9])`.

### 1️⃣ Trigonometric ufunc: `np.sin`

```python
np.sin(v1)
```

```
[[ 0.90929743  0.14112001 -0.95892427]
 [ 0.6569866  -0.99999021  0.42016704]
 [-0.96139749  0.14987721 -0.8462204 ]]
```

### 2️⃣ Arithmetic ufunc: `np.add`

```python
q = np.array([7, 8, 9])
np.add(v1, q)
```

```
[[ 9 11 14]
 [14 19 22]
 [24 27 32]]
```

`q` is added to every row of `v1` (this is called *broadcasting*).

### 3️⃣ Comparison ufuncs: `np.greater`, `np.equal`, `np.not_equal`

<details>
<summary><b>🔍 Show code and output</b></summary>

```python
np.greater(v1, q)
```
```
[[False False False]
 [False  True  True]
 [ True  True  True]]
```

```python
np.equal(v1, q)
```
```
[[False False False]
 [ True False False]
 [False False False]]
```

```python
np.not_equal(v1, q)
```
```
[[ True  True  True]
 [False  True  True]
 [ True  True  True]]
```

</details>

---

## 🚀 Bonus Challenges

<details>
<summary><b>Try other ufuncs (click to expand)</b></summary>

| Function | What it does |
|---|---|
| `np.sqrt(v1)` | Square root of each element |
| `np.exp(v1)` | e raised to each element |
| `np.square(v1)` | Square of each element |
| `np.log(v1)` | Natural log of each element |
| `np.multiply(v1, q)` | Element-wise product with broadcasting |

</details>

---

## 🗂️ Files in this Module

| File | Description |
|---|---|
| `Module_1.ipynb` | Main notebook with vector operations and ufuncs |
| `Module_1_worksheet.ipynb` | Practice worksheet |

---

## ✅ Submission Tips

1. Run **Kernel → Restart & Run All** so the saved outputs come from a clean run.
2. Commit and push to GitHub:

```bash
git add .
git commit -m "Update Module 1 README"
git push origin main
```



