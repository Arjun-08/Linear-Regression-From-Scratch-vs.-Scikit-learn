# Linear Regression: From Scratch vs. Scikit-learn

## Problem Statement

Develop a linear regression model to estimate the **average number of rainy days in a month** as a function of **average temperature**, following the linear regression algorithm developed in class.

The requirements are:

1. Develop the linear regression model using the **Least Square Estimation (LSE)** method.
2. Consider **80% of the data for training** and the remaining **20% for testing**, using random sampling.
3. Report the **Mean Squared Error (MSE)** on the test set.
4. Use the `sklearn` package in Python to construct the same linear regression model.
5. Compare the coefficients and MSE obtained using the manual implementation and `sklearn`.

---

## Dataset

The dataset consists of average temperature and average rainy days.

| Average Temperature | Average Rainy Days | Average Temperature | Average Rainy Days |
| ------------------: | -----------------: | ------------------: | -----------------: |
|                 0.2 |               12.1 |                16.1 |               12.1 |
|                 1.6 |               11.1 |                20.1 |                8.4 |
|                 5.7 |               10.7 |                23.5 |                3.4 |
|                11.3 |               11.0 |                23.4 |                2.6 |
|                18.8 |                4.0 |                12.9 |                6.8 |

The complete set of observations is:

| Average Temperature (`x`) | Average Rainy Days (`y`) |
| ------------------------: | -----------------------: |
|                       0.2 |                     12.1 |
|                       1.6 |                     11.1 |
|                       5.7 |                     10.7 |
|                      11.3 |                     11.0 |
|                      18.8 |                      4.0 |
|                      16.1 |                     12.1 |
|                      20.1 |                      8.4 |
|                      23.5 |                      3.4 |
|                      23.4 |                      2.6 |
|                      12.9 |                      6.8 |

---

# 1. Linear Regression Using Least Square Estimation

We can find `β₀` and `β₁` using the **Least Square Estimation (LSE)** method.

The linear regression model is:

```text
ŷ = β₀ + β₁x
```

where:

* `ŷ` = predicted number of rainy days
* `x` = average temperature
* `β₀` = intercept
* `β₁` = slope

---

## 1.1 Train-Test Split

80% of the data is used as the **training set**, while 20% is used as the **test set** using random sampling.

### Training Set

The following 8 `(x, y)` pairs are used for training:

```text
(16.1, 12.1)
(0.2, 12.1)
(23.5, 3.4)
(5.7, 10.7)
(12.9, 6.8)
(18.8, 4.0)
(11.3, 11.0)
(20.1, 8.4)
```

### Test Set

The remaining 2 `(x, y)` pairs are used for testing:

```text
(23.4, 2.6)
(1.6, 11.1)
```

---

# 2. LSE Objective Function

The objective of Least Square Estimation is to minimize the sum of squared errors:

```text
S(β₀, β₁) = Σᵢ₌₁ᴺ (yᵢ − β₀ − β₁xᵢ)²
```

Taking the partial derivative with respect to `β₀`:

```text
∂S/∂β₀ = −2 Σᵢ (yᵢ − β₀ − β₁xᵢ) = 0
```

Therefore:

```text
Σᵢ yᵢ − Nβ₀ − β₁Σᵢ xᵢ = 0
```

This gives:

```text
(A)
```

Taking the partial derivative with respect to `β₁`:

```text
∂S/∂β₁ = −2 Σᵢ xᵢ(yᵢ − β₀ − β₁xᵢ) = 0
```

Therefore:

```text
Σᵢ xᵢyᵢ − β₀Σᵢ xᵢ − β₁Σᵢ xᵢ² = 0
```

This gives:

```text
(B)
```

---

# 3. Derivation of β₀ and β₁

From equation `(A)`:

```text
β₀ = ȳ − β₁x̄
```

This is:

```text
(C)
```

Substituting equation `(C)` into equation `(B)`:

```text
Σᵢ xᵢyᵢ − (ȳ − β₁x̄)Σᵢ xᵢ − β₁Σᵢ xᵢ² = 0
```

Since:

```text
Σᵢ xᵢ = Nx̄
```

we obtain:

```text
Σᵢ xᵢyᵢ − Nx̄ȳ + Nβ₁x̄² − β₁Σᵢ xᵢ² = 0
```

Rearranging:

```text
Σᵢ xᵢyᵢ − Nx̄ȳ
=
β₁(Σᵢ xᵢ² − Nx̄²)
```

Using the identities:

```text
Σᵢ (xᵢ − x̄)(yᵢ − ȳ)
=
Σᵢ xᵢyᵢ − Nx̄ȳ
```

and:

```text
Σᵢ (xᵢ − x̄)²
=
Σᵢ xᵢ² − Nx̄²
```

we obtain the least-squares estimate for the slope:

```text
β̂₁ =
Σᵢ₌₁ᴺ (xᵢ − x̄)(yᵢ − ȳ)
--------------------------------
Σᵢ₌₁ᴺ (xᵢ − x̄)²
```

Finally, the intercept is:

```text
β̂₀ = ȳ − β̂₁x̄
```

---

# 4. Training Data Calculations

### Training `x` values

```text
[16.1, 0.2, 23.5, 5.7, 12.9, 18.8, 11.3, 20.1]
```

### Training `y` values

```text
[12.1, 12.1, 3.4, 10.7, 6.8, 4.0, 11.0, 8.4]
```

### Required Summations

| Quantity |   Value |
| -------- | ------: |
| `Σxᵢ`    |   108.6 |
| `Σyᵢ`    |    68.5 |
| `Σxᵢyᵢ`  |  794.18 |
| `Σxᵢ²`   | 1895.54 |
| `N`      |       8 |

---

## 4.1 Calculate the Means

The mean temperature is:

```text
x̄ = Σxᵢ / N

x̄ = 108.6 / 8

x̄ = 13.575
```

The mean number of rainy days is:

```text
ȳ = Σyᵢ / N

ȳ = 68.5 / 8

ȳ = 8.5625
```

Therefore:

```text
x̄ = 13.575
ȳ = 8.5625
```

---

# 5. Calculate the Slope β₁

Using:

```text
β̂₁ =
(Σxᵢyᵢ − Nx̄ȳ)
----------------
(Σxᵢ² − Nx̄²)
```

### Numerator

```text
Σxᵢyᵢ − Nx̄ȳ

= 794.18 − 8(13.575)(8.5625)

= −135.7075
```

### Denominator

```text
Σxᵢ² − Nx̄²

= 1895.54 − 8(13.575²)

= 421.295
```

Therefore:

```text
β̂₁ = −135.7075 / 421.295

β̂₁ ≈ −0.322119892237031
```

Thus, the estimated slope is:

```text
β̂₁ ≈ −0.322119892237031
```

---

# 6. Calculate the Intercept β₀

Using:

```text
β̂₀ = ȳ − β̂₁x̄
```

we get:

```text
β̂₀
= 8.5625 − (−0.322119892237031)(13.575)

≈ 12.935277537117695
```

Therefore:

```text
β̂₀ ≈ 12.935277537117695
```

---

# 7. Final Linear Regression Model

The resulting linear regression model is:

```text
ŷ = 12.935277537117695 − 0.322119892237031x
```

This indicates a negative relationship between average temperature and average rainy days for this dataset.

The fitted regression line can be represented as:

```text
Rainy Days = 12.935277537117695 − 0.322119892237031 × Temperature
```

---

# 8. Test Set Prediction

The test set contains:

```text
(23.4, 2.6)
(1.6, 11.1)
```

---

## 8.1 Test Point 1

For:

```text
x = 23.4
y = 2.6
```

The predicted value is:

```text
ŷ = 12.935277537117695
     − 0.322119892237031(23.4)

ŷ ≈ 5.397672055900183
```

Therefore, the prediction error is:

```text
e = y − ŷ

e = 2.6 − 5.39767206

e ≈ −2.79767206
```

---

## 8.2 Test Point 2

For:

```text
x = 1.6
y = 11.1
```

The predicted value is:

```text
ŷ = 12.935277537117695
     − 0.322119892237031(1.6)

ŷ ≈ 12.419885707335207
```

Therefore, the prediction error is:

```text
e = y − ŷ

e = 11.1 − 12.41988571

e ≈ −1.31988571
```

---

# 9. Test Mean Squared Error

The Mean Squared Error is calculated as:

```text
MSE = (1/N) Σᵢ (yᵢ − ŷᵢ)²
```

For the two test samples:

```text
MSE =
[(-2.79767206)² + (-1.31988571)²] / 2
```

Therefore:

```text
MSE ≈ 4.784533617336366
```

### Manual LSE Result

```text
Test MSE ≈ 4.784533617336366
```

---

# 10. Linear Regression Using Scikit-learn

The same linear regression model is constructed using the `sklearn` package in Python.

The `LinearRegression` model is fitted with:

```text
fit_intercept = True
```

The model is trained using exactly the same training data and evaluated on the same test data.

The resulting coefficients are:

```text
sklearn coefficient (slope)
= −0.322119892237031
```

```text
sklearn intercept
= 12.935277537117695
```

The resulting test MSE is:

```text
Test MSE ≈ 4.784533617336366
```

---

# 11. Comparison: Manual LSE vs. Scikit-learn

| Method       | β₀ (Intercept) | β₁ (Slope) | Test MSE |
| ------------ | -------------: | ---------: | -------: |
| Manual LSE   |      12.935278 |  -0.322120 | 4.784534 |
| Scikit-learn |      12.935278 |  -0.322120 | 4.784534 |

The results are identical up to numerical precision.

---

# 12. Conclusion

The linear regression model was first developed manually using the **Least Square Estimation (LSE)** method. The resulting model was:

```text
ŷ = 12.935277537117695 − 0.322119892237031x
```

When evaluated on the 20% test set, the manually implemented model achieved:

```text
Test MSE = 4.784533617336366
```

The same model was then constructed using `sklearn.linear_model.LinearRegression` with `fit_intercept=True`.

The Scikit-learn implementation produced the same slope and intercept:

```text
β₀ = 12.935277537117695
β₁ = −0.322119892237031
```

and the same test MSE:

```text
Test MSE = 4.784533617336366
```

Therefore, the manual implementation and the Scikit-learn implementation give **identical results**, confirming the correctness of the Least Square Estimation implementation.

---

# 13. Implementation

The repository contains both implementations:

```text
Manual LSE
    ↓
Calculate β₀ and β₁
    ↓
Construct regression model
    ↓
Predict test samples
    ↓
Calculate MSE
    ↓
Compare with Scikit-learn
```

The Scikit-learn implementation uses:

```python
from sklearn.linear_model import LinearRegression
```

and the model is configured with:

```python
LinearRegression(fit_intercept=True)
```

---

## Results Summary

| Metric           | Manual LSE | Scikit-learn |
| ---------------- | ---------: | -----------: |
| Intercept (`β₀`) |  12.935278 |    12.935278 |
| Slope (`β₁`)     |  -0.322120 |    -0.322120 |
| Test MSE         |   4.784534 |     4.784534 |
| Result           |  Identical |    Identical |

**Final observation:** The from-scratch Least Square Estimation implementation reproduces the results obtained from Scikit-learn, demonstrating that the mathematical derivation and implementation are consistent with the standard linear regression implementation.
