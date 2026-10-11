# Module 4: Regularized Cost and Gradient

This module extends the standard **linear** and **logistic** regression cost and gradient functions with an **L2 regularization** term. Regularization discourages large parameter values, which helps reduce overfitting.

The work is in the notebook `Module_4_Regularization_Solution.ipynb`.

## What's Covered

- Regularized cost function for **linear regression**
- Regularized cost function for **logistic regression**
- Regularized gradient for **linear regression**
- Regularized gradient for **logistic regression**
- A numerically stable `sigmoid` helper

## Key Equations

**Regularized linear regression cost**

$$J(\mathbf{w},b) = \frac{1}{2m} \sum_{i=0}^{m-1} (h(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

**Regularized logistic regression cost**

$$J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ -y^{(i)} \log(h(\mathbf{x}^{(i)})) - (1 - y^{(i)}) \log(1 - h(\mathbf{x}^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

**Regularized gradients (linear and logistic)**

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (h(\mathbf{x}^{(i)}) - y^{(i)})\,x_j^{(i)} + \frac{\lambda}{m} w_j \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (h(\mathbf{x}^{(i)}) - y^{(i)})$$

The two models differ only in how `h(x)` is computed:

| Model    | h(x)                                  |
|----------|---------------------------------------|
| Linear   | `w · x + b`                           |
| Logistic | `sigmoid(w · x + b)`                  |

The bias term `b` is **not** regularized, which is standard practice.

## Functions Implemented

| Function                         | Purpose                                         |
|----------------------------------|-------------------------------------------------|
| `sigmoid(z)`                     | Sigmoid with clipping to avoid overflow         |
| `compute_cost_linear_reg`        | Regularized cost for linear regression          |
| `compute_cost_logistic_reg`      | Regularized cost for logistic regression        |
| `compute_gradient_linear_reg`    | Regularized gradients for linear regression     |
| `compute_gradient_logistic_reg`  | Regularized gradients for logistic regression   |

## Expected Outputs

Each function is checked against a small test case (`np.random.seed(1)`).

| Test                               | Expected result                                                  |
|------------------------------------|------------------------------------------------------------------|
| Linear regularized cost            | `0.07917239320214275`                                            |
| Logistic regularized cost          | `0.6850849138741673`                                             |
| Linear gradient `dj_db`            | `0.6648774569425726`                                             |
| Linear gradient `dj_dw`            | `[0.29653214748822276, 0.4911679625918033, 0.21645877535865857]` |
| Logistic gradient `dj_db`          | `0.341798994972791`                                              |
| Logistic gradient `dj_dw`          | `[0.17380012933994293, 0.32007507881566943, 0.10776313396851499]`|

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

```bash
pip install numpy matplotlib jupyterlab
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/arman-yusufzai/unified-mentor-training.git
   cd unified-mentor-training/Module_4
   ```
2. Launch Jupyter:
   ```bash
   jupyter lab
   ```
3. Open `Module_4_Regularization_Solution.ipynb` and run all cells from top to bottom.

## Folder Contents

```
Module_4/
├── Module_4.ipynb
├── Module_4_Regularization.ipynb
├── Module_4_Regularization_Solution.ipynb
└── README.md
```

## Author

Arman Yusufzai, Unified Mentor training
