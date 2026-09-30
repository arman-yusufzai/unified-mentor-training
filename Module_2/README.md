# 📊 Module 2 — Cost Function & Gradient Descent

This module focuses on **Linear Regression**, **Cost Function**, and **Gradient Descent** using Python and Jupyter Notebook.

The objective is to build a simple model that can predict **housing prices based on house size** and understand how machine learning parameters are optimized during training.

---

## 📌 Problem Statement

The model is trained using a small housing dataset containing:

| Size (1000 sqft) | Price (INR in Lakhs) |
| ---------------: | -------------------: |
|                1 |                   30 |
|                2 |                   50 |
|                3 |                   68 |
|                4 |                   75 |
|                5 |                  100 |

The goal is to learn a linear relationship between house size and price.

---

## 🧠 Concepts Covered

### 1. Linear Regression

The model uses a linear hypothesis:

$$
h(x) = wx + b
$$

Where:

* `x` → input feature (house size)
* `w` → weight
* `b` → bias
* `h(x)` → predicted price

---

### 2. Cost Function

The cost function measures how well the model's predictions match the actual housing prices.

$$
J(w,b)=
\frac{1}{2m}
\sum_{i=0}^{m-1}
(h(x^{(i)})-y^{(i)})^2
$$

This is based on the **Mean Squared Error (MSE)**.

A lower cost indicates that the model's predictions are closer to the actual values.

---

### 3. Gradient Descent

Gradient Descent is used to find suitable values of `w` and `b` by minimizing the cost function.

The parameters are updated using:

$$
w = w-\alpha\frac{\partial J(w,b)}{\partial w}
$$

$$
b = b-\alpha\frac{\partial J(w,b)}{\partial b}
$$

Where:

* `α` → learning rate
* `∂J/∂w` → gradient with respect to weight
* `∂J/∂b` → gradient with respect to bias

The process is repeated until the model approaches convergence.

---

## 📐 Gradient Equations

The notebook uses the following partial derivatives:

$$
\frac{\partial J(w,b)}{\partial w}
=
\frac{1}{m}
\sum_{i=0}^{m-1}
(h(x^{(i)})-y^{(i)})x^{(i)}
$$

$$
\frac{\partial J(w,b)}{\partial b}
=
\frac{1}{m}
\sum_{i=0}^{m-1}
(h(x^{(i)})-y^{(i)})
$$

These gradients determine how the parameters should be changed to reduce the cost.

---

## 🔄 Machine Learning Workflow

```text
Training Data
     ↓
Linear Regression Model
     ↓
Initial w and b
     ↓
Prediction
     ↓
Calculate Cost
     ↓
Calculate Gradients
     ↓
Update w and b
     ↓
Repeat Gradient Descent
     ↓
Cost Decreases
     ↓
Optimal Parameters
     ↓
Predict Housing Prices
```

---

## 📈 Cost vs Iterations

The notebook visualizes the **cost against the number of iterations**.

A successful Gradient Descent run should show the cost decreasing as training progresses.

This visualization helps understand whether the model is successfully approaching a lower-cost solution.

---

## 🔮 Predictions

After finding suitable values of `w` and `b`, the trained model is used to predict housing prices for different house sizes.

The notebook demonstrates predictions for:

* 1000 sqft
* 2000 sqft
* 1750 sqft

The prediction is calculated using:

$$
h(x)=wx+b
$$

---

## 🛠️ Technologies Used

* 🐍 Python
* 📓 Jupyter Notebook
* 🔢 NumPy
* 📊 Matplotlib

---

## 📁 Files

```text
Module_2/
│
├── .gitignore
├── Module_2.ipynb
├── UM_Module_2_Worksheet.ipynb
└── README.md
```

### `Module_2.ipynb`

Completed solution notebook containing the implementation and results.

### `UM_Module_2_Worksheet.ipynb`

Original worksheet provided for the module.

### `README.md`

Documentation and explanation of the module.

---

## 🎯 Learning Objectives

After completing this module, you should understand:

* What Linear Regression is
* How a prediction is generated using `w` and `b`
* What a Cost Function represents
* How Mean Squared Error is used
* What a gradient represents
* How Gradient Descent updates model parameters
* The role of the learning rate
* How iterations improve the model
* How to visualize cost reduction
* How a trained Linear Regression model makes predictions

---

## 🚀 How to Run

Open the notebook using Jupyter Notebook or JupyterLab.

```bash
jupyter notebook
```

Then open:

```text
Module_2.ipynb
```

Run the notebook cells sequentially to reproduce the calculations, Gradient Descent process, visualizations, and predictions.

---

## 📚 Module Summary

**Module 2** demonstrates the fundamental training process of a Linear Regression model:

> **Prediction → Cost Calculation → Gradient Calculation → Parameter Update → Repeat → Prediction**

This module provides the mathematical and programming foundation for understanding how basic machine learning models learn from data.
