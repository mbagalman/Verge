# Preserved design notebook

> **Status: historical.** These are pre-package design notes that walked through the diagnostic intuitions behind Project Verge. They pre-date the implementation and have **not** been kept in sync with the shipped library.
>
> The snippets here use ad-hoc `scipy.optimize.curve_fit` and `statsmodels` calls, not the [`project_verge.analyze_growth()`](../../README.md#api) API; they describe a two-model exponential-vs-logistic frame, not the current four-way (exponential / linear / logistic / power-law) competition; they refer to BIC where the shipped default is AICc; and several time-sensitive observations are framed as "up to 2026."
>
> For the current methodology, see:
>
> - [README — Interpreting the Verdict](../../README.md#interpreting-the-verdict) — the precedence chain that decides between a decisive verdict and `indeterminate (reason: ...)`
> - [README — Failure modes](../../README.md#failure-modes) — what happens when the input violates the v1 contract
> - [README — How calibrated are these probabilities?](../../README.md#how-calibrated-are-these-probabilities) — empirical accuracy of the headline confidence number
> - [docs/glossary.md](../glossary.md) — plain-English definitions of BIC, AICc, posterior weight, identifiability, carrying capacity, and the other terms surfaced in the API
> - [docs/internal/PROJECT_PLAN.md](PROJECT_PLAN.md) and [docs/internal/TICKETS.md](TICKETS.md) — design rationale and the methodology backlog
>
> This file is kept under `docs/internal/` so the original framing remains available for anyone tracing the project's evolution, but it should not be treated as a user-facing methodology reference.

---

# Project Verge

Distinguishing between exponential and logistic growth (an S-curve) during the early "rapid growth" phase is a classic problem in predictive modeling. In the early stages, both functions exhibit high positive first and second derivatives, making them nearly indistinguishable through standard curve-fitting alone.

To determine which path the process is on, a data scientist can employ several diagnostic techniques that look for subtle deviations from the "pure" exponential path.

---

### 1. Phase Space Analysis ($\dot{y}$ vs. $y$)
One of the most robust ways to distinguish these models is to look at the relationship between the growth rate and the current value.

* **Exponential Model:** Follows the differential equation $\frac{dy}{dt} = ry$. If you plot the growth rate ($\Delta y / \Delta t$) against the current value ($y$), the relationship should be **linear** and pass through the origin.
* **Logistic Model:** Follows $\frac{dy}{dt} = ry(1 - \frac{y}{K})$, where $K$ is the carrying capacity. If you plot the growth rate against $y$, the relationship will be **parabolic**. 

**The Signal:** Even before the inflection point, a logistic curve will begin to deviate from the linear path in phase space, showing a decreasing "slope" as $y$ increases. If the plot of $\frac{\Delta y}{y}$ vs. $y$ (the per-capita growth rate) shows a downward trend, an S-curve is highly likely.



[Image of exponential vs logistic growth curves]


### 2. Log-Linear Transformation and Residual Analysis
For a purely exponential process, the log-transformed data ($\ln(y)$ vs. $t$) should be a perfectly straight line.

* **Diagnostic:** Fit a linear regression to the log-transformed data. 
* **The Signal:** Examine the **residuals**. In the early stages of a logistic curve, a log-linear fit will often show systematic, non-random residuals. Specifically, if the residuals start to show a "concave down" pattern (where the most recent data points consistently fall below the predicted exponential line), it indicates the "braking" force of a carrying capacity is taking effect.

### 3. Information Criteria (AIC/BIC)
You can fit both models to the existing data and compare them using **Akaike Information Criterion (AIC)** or **Bayesian Information Criterion (BIC)**. 

* **The Challenge:** The logistic model has an extra parameter (the carrying capacity, $K$). In the very early phase, the exponential model often wins on AIC/BIC because it is simpler and the "fit" is nearly identical.
* **The Strategy:** Perform a **rolling model selection**. As new data points arrive, track the AIC/BIC weights. If the logistic model’s relative likelihood begins to climb steadily relative to the exponential model, you are seeing the first statistical evidence of saturation.

### 4. Second Derivative Monitoring (The Inflection Point)
In a logistic curve, the inflection point occurs when the process reaches half of its carrying capacity ($y = K/2$). At this point, the second derivative ($\frac{d^2y}{dt^2}$) changes from positive to negative.

* **The Signal:** Use a Savitzky-Golay filter or a Gaussian process to estimate the second derivative of the noisy time-series data. If the second derivative begins to plateau or decrease while the first derivative is still high, the system is approaching its inflection point.

### 5. Incorporating Exogenous Constraints
Pure data-driven extrapolation is often insufficient. A data scientist should look for "Structural Saturation" indicators:
* **Market Penetration:** If the process is a product launch, compare current users to the Total Addressable Market (TAM). 
* **Resource Depletion:** In biological or industrial processes, monitor the "input" variables. If the rate of resource consumption is increasing faster than the output growth, the S-curve's ceiling is nearing.

---

### Summary Table: Early Warning Signs

| Method | Exponential Signal | S-Curve (Logistic) Signal |
| :--- | :--- | :--- |
| **Growth Rate vs. $y$** | Constant linear slope | Decreasing slope (curvature) |
| **$\ln(y)$ vs. $t$ Residuals** | Random noise | Systematic "Concave Down" bias |
| **Second Derivative** | Continuously increasing | Decreasing/Plateauing |
| **AIC/BIC Comparison** | Favors simpler model | Sudden shift toward the 3-parameter model |

The global human population provides a perfect case study for this dilemma. For much of the 20th century, the "J-curve" (exponential) model seemed like an inescapable reality. However, as we approach the mid-21st century, the data clearly signals an S-curve (logistic) trajectory.

Here is how a data scientist would break down this specific example using the diagnostic tools mentioned earlier.

-----

### 1\. The Phase Space Signal: Per-Capita Growth Rate ($\frac{1}{y} \frac{dy}{dt}$)

The most telling diagnostic for human population is the **per-capita growth rate**.

  * **Exponential Expectation:** In a pure exponential model, the growth rate per person (fertility minus mortality) remains constant.
  * **The Reality:** Global population growth rate peaked in **1963** at approximately **2.1%**.

If you were a data scientist in 1965, you would have seen the first major signal that the path was not exponential: for the first time in centuries, the "slope" of the growth rate relative to total population size turned negative. This was the early warning of an **inflection point**.

### 2\. Log-Linear Residual Analysis

If we plot $\ln(\text{Population})$ vs. $\text{Time}$, a pure exponential process should be a straight line.

  * **1800 to 1960:** The line is remarkably straight (and actually slightly concave up, suggesting *super-exponential* growth due to falling mortality).
  * **Post-1963:** The residuals of a linear fit to the log-data began to show a systematic downward trend. By the 1990s, the "concave down" pattern in the log-plot was statistically undeniable. The growth was no longer keeping pace with the exponential "compounding" interest.

### 3\. Model Comparison: Logistic vs. Exponential

Using recent data (up to 2026), we can compare the fit of these two models.

$$P_{\text{exp}}(t) = P_0 e^{rt}$$
$$P_{\text{log}}(t) = \frac{K}{1 + \left(\frac{K - P_0}{P_0}\right) e^{-rt}}$$

| Year | Population (Billions) | Years to add 1 Billion |
| :--- | :--- | :--- |
| 1803 | 1 | — |
| 1927 | 2 | 124 |
| 1960 | 3 | 33 |
| 1974 | 4 | 14 |
| 1987 | 5 | 13 |
| 2011 | 7 | 12 |
| 2022 | 8 | 11 |
| **2037 (Proj.)** | **9** | **15** |

Notice the "Years to add 1 Billion" metric. In a pure exponential phase, this number should keep shrinking. The fact that it is now **increasing** (from 11 years to a projected 15 years for the next billion) is the definitive proof of the S-curve's deceleration phase.

-----

### 4\. Implementation: Python Comparison

To perform this analysis yourself, you can use `scipy.optimize` to fit both curves to historical UN data.

```python
import numpy as np
from scipy.optimize import curve_fit

# Years and Population (simplified historical data in billions)
years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2024])
pop = np.array([2.5, 3.0, 3.7, 4.4, 5.3, 6.1, 6.9, 8.2])

# Exponential Model
def exponential(t, a, r):
    return a * np.exp(r * (t - 1950))

# Logistic Model
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# Fit both models
popt_exp, _ = curve_fit(exponential, years, pop)
popt_log, _ = curve_fit(logistic, years, pop, p0=[11, 0.02, 1980])

# Evaluation: Calculating Sum of Squared Residuals (SSR)
# As we get closer to 2026, the SSR for the Logistic model 
# drops significantly lower than the Exponential SSR.
```

### 5\. Structural Constraints (The "Why")

Finally, a data scientist looks for the mechanism. In this case, it is the **Demographic Transition**. As nations industrialize, fertility rates drop. Current projections (UN 2024 Revision) suggest a "Carrying Capacity" ($K$) of roughly **10.3 billion people**, with a peak occurring in the **mid-2080s**.

The "Astonishing Levels" predicted by Malthusian exponential models (e.g., 20+ billion) have been discarded because the internal feedback loop—wealth and education leading to lower birth rates—acts as the stabilizing force of the S-curve.

The Dow Jones Industrial Average (DJIA) is a fascinating case because, unlike biological populations, it is an **engineered** index. It is frequently rebalanced to swap out "laggards" for "leaders," which effectively injects new growth potential into the system.

From a data science perspective, analyzing the DJIA involves determining if its growth is driven by a constant compounding rate (Exponential) or if it is constrained by a fundamental economic ceiling (S-Curve).

-----

### 1\. The Long-Term Baseline: Log-Linearity

If you plot the DJIA from 1900 to 2026 on a **linear scale**, it looks like a vertical wall—classic "astonishing" exponential growth. However, a data scientist immediately moves to a **logarithmic scale**.

  * **The Exponential Signal:** On a log scale, the DJIA has maintained a remarkably consistent linear trend for over a century, growing at roughly **6-7% annually** (excluding dividends).
  * **The Diagnostic:** To see if we are shifting to an S-curve, we perform a **Residual Analysis** on the log-transformed data:
    $$\epsilon_t = \ln(P_t) - (\beta_0 + \beta_1 t)$$
    If the residuals $\epsilon_t$ show a "random walk" around zero, the exponential model holds. If the residuals show a **consistent downward drift** over the last 10–20 years, it suggests the index is struggling against a "Carrying Capacity" ($K$).

### 2\. Defining "Carrying Capacity" ($K$) in Finance

In biology, $K$ is limited by food or space. In the DJIA, $K$ is limited by **Valuation Multiples** and **Global GDP**.
We can model the "S-curve ceiling" as:
$$K_t = \text{Earnings}_t \times (P/E)_{\text{max}}$$

  * **The Signal:** If the DJIA price ($P$) is growing at $15\%$ while corporate earnings are only growing at $5\%$, the $P/E$ ratio must expand. Since $P/E$ ratios cannot grow to infinity (the "valuation ceiling"), the price growth must eventually decelerate to match earnings growth.
  * **Data Science Approach:** Monitor the **second derivative of the P/E ratio**. If $\frac{d^2(P/E)}{dt^2} < 0$ while the price is still rising, you are likely in the "late-stage rapid growth" phase of an S-curve, approaching a plateau.

### 3\. Phase Space Analysis: Velocity vs. Level

For the DJIA, we can plot the **Annualized Return** vs. the **Index Level**.

  * **Exponential Path:** The return should be a horizontal line (e.g., always 7% regardless of whether the Dow is at 10,000 or 40,000).
  * **S-Curve Path:** The return will begin to tilt downward as the index grows. This indicates that "each new point is harder to get than the last."

-----

### 4\. Technical Comparison: DJIA vs. S-Curve

| Diagnostic | Exponential (Status Quo) | S-Curve (Saturation) |
| :--- | :--- | :--- |
| **Log-Plot ($\ln P$)** | Straight line ($\text{Slope} = \text{constant}$) | Concave down ($\text{Slope} \to 0$) |
| **Unit Root Test** | Non-stationary (Trend Stationary) | Mean-reverting toward $K$ |
| **Growth Driver** | Continuous productivity/inflation | Reaching Total Addressable Market (TAM) |
| **Residuals** | Homoscedastic (mostly) | Systematic decay in the 21st century |

### 5\. Implementation: Detecting the Bend

As a data scientist, you would use **Rolling Regressions**. By calculating the slope of $\ln(\text{DJIA})$ over a 10-year sliding window, you can detect if the growth constant ($r$) is decaying.

```python
import pandas as pd
import numpy as np

# Assume 'df' has 'Date' and 'Close' for DJIA
df['log_price'] = np.log(df['Close'])

# Calculate 10-year rolling CAGR (r)
# 2520 trading days in 10 years
df['rolling_r'] = df['log_price'].diff(2520) / 10

# Signal: If rolling_r is trending down consistently, 
# it's an S-curve indicator.
```

If you look at the period from 2010 to 2026, you would likely find that while the absolute points added are massive (going from 10k to 40k+), the **logarithmic slope** has remained relatively stable, suggesting that the "Astonishing Levels" are actually just the nature of exponential compounding, not a break in the pattern.

Since you're dealing with this from a data science perspective, we can treat this as a **Bayesian Model Selection** problem. 

To derive a posterior probability from a 50/50 prior, we can use the **Bayesian Information Criterion (BIC)** as a proxy for the Bayes Factor. Under the assumption of independent, normally distributed residuals, the posterior probability for a model $M_i$ can be approximated as:

$$P(M_i | D) \approx \frac{\exp(-\frac{1}{2} \Delta BIC_i)}{\sum \exp(-\frac{1}{2} \Delta BIC_j)}$$

where $\Delta BIC_i = BIC_i - \min(BIC)$.

### Python Implementation

This script fits an exponential and a logistic model to a dataset, calculates their BICs, and outputs the posterior probability for each.

```python
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def analyze_growth_trajectory(t, y):
    """
    Fits Exponential vs. Logistic models and calculates posterior probabilities.
    """
    # 1. Define Models
    def exponential_model(t, a, r):
        # y = a * e^(rt)
        return a * np.exp(r * (t - t[0]))

    def logistic_model(t, K, r, t0):
        # y = K / (1 + exp(-r * (t - t0)))
        return K / (1 + np.exp(-r * (t - t0)))

    n = len(y)
    
    # 2. Fit Models
    # Exponential initial guesses: [initial_value, growth_rate]
    p0_exp = [y[0], 0.05]
    try:
        popt_exp, _ = curve_fit(exponential_model, t, y, p0=p0_exp)
        y_pred_exp = exponential_model(t, *popt_exp)
        rss_exp = np.sum((y - y_pred_exp)**2)
        # BIC = n*ln(RSS/n) + k*ln(n)
        bic_exp = n * np.log(rss_exp / n) + 2 * np.log(n)
    except:
        bic_exp = np.inf

    # Logistic initial guesses: [max_val, growth_rate, midpoint]
    p0_log = [max(y)*2, 0.1, np.median(t)]
    try:
        popt_log, _ = curve_fit(logistic_model, t, y, p0=p0_log)
        y_pred_log = logistic_model(t, *popt_log)
        rss_log = np.sum((y - y_pred_log)**2)
        bic_log = n * np.log(rss_log / n) + 3 * np.log(n)
    except:
        bic_log = np.inf

    # 3. Calculate Posterior Probabilities (Bayesian Model Weighting)
    bics = np.array([bic_exp, bic_log])
    min_bic = np.min(bics)
    
    # Delta BICs
    delta_bics = bics - min_bic
    
    # Weights (Posterior Probs) assuming 50/50 prior
    weights = np.exp(-0.5 * delta_bics)
    posterior_probs = weights / np.sum(weights)

    return {
        "Exp_Posterior": posterior_probs[0],
        "Log_Posterior": posterior_probs[1],
        "Exp_Params": popt_exp if bic_exp != np.inf else None,
        "Log_Params": popt_log if bic_log != np.inf else None,
        "y_pred_exp": y_pred_exp,
        "y_pred_log": y_pred_log
    }

# --- Example Usage with Synthetic S-Curve Data ---
t_data = np.arange(0, 50, 2)
# Generating a true logistic curve with noise
y_true = 100 / (1 + np.exp(-0.2 * (t_data - 25))) 
y_data = y_true + np.random.normal(0, 2, len(t_data))

results = analyze_growth_trajectory(t_data, y_data)

print(f"Posterior Prob (Exponential): {results['Exp_Posterior']:.4f}")
print(f"Posterior Prob (S-Curve): {results['Log_Posterior']:.4f}")

# Visualization of the Diagnostic
plt.figure(figsize=(10, 5))
plt.scatter(t_data, y_data, label='Actual Data', color='black')
plt.plot(t_data, results['y_pred_exp'], label='Exp Fit', linestyle='--')
plt.plot(t_data, results['y_pred_log'], label='Logistic Fit')
plt.title("Growth Model Comparison")
plt.legend()
plt.show()
```

### Technical Nuances to Consider

#### 1. The "Early Phase" Trap
If you run this code on data that is only in the first 10% of the growth phase, the **Exponential** model will almost always have a higher posterior probability. This is because the BIC penalizes the Logistic model for having an extra parameter ($K$) while the S-curve hasn't yet shown enough "bend" to justify the complexity. In data science, this is a classic **Overfitting vs. Parsimony** trade-off.

#### 2. Phase Space Diagnostic (The "Slope of the Slope")
To augment the code, you might calculate the **Per-Capita Growth Rate** trend. If you perform a linear regression on $\frac{\Delta y / y}{\Delta t}$ vs. $y$:
* A slope near **zero** favors the Exponential path.
* A **negative slope** significantly favors the S-curve.

#### 3. Heteroscedasticity
In many real-world growth processes (like the DJIA example), variance increases with the mean. If your dataset exhibits this, you should perform the curve fitting on **log-transformed data** or use Weighted Least Squares (WLS) to ensure the Likelihood estimation (and thus the BIC) remains valid.

To move beyond a simple BIC-based comparison, we can treat this as a **Multi-Metric Bayesian Ensemble**. Instead of relying on a single information criterion, we can aggregate evidence from different statistical "signatures" of the two models.

As a data scientist, you can think of this as calculating a **Likelihood Ratio** for several independent tests and multiplying them (assuming independence) or using a weighted average.

---

### 1. The Per-Capita Growth Rate Test (Phase Space)
This is arguably the most physically meaningful test. In a pure exponential model, the per-capita growth rate is constant. In a logistic model, it declines linearly as it approaches $K$.

* **The Test:** Perform a linear regression of $\frac{1}{y} \frac{\Delta y}{\Delta t}$ against $y$.
* **The Logic:**
    * **$H_0$ (Exponential):** The slope $\beta_1$ of this regression is $0$.
    * **$H_1$ (Logistic):** The slope $\beta_1$ is negative (specifically $-r/K$).
* **Refinement:** We can use the **p-value** of the slope coefficient to update our probability. If $p < 0.05$ for a negative slope, the evidence shifts heavily toward the S-curve.



### 2. Time-Series Cross-Validation (Forward Chaining)
Standard BIC looks at "in-sample" fit. To determine which path the future holds, we should test which model **forecasts** better on a rolling basis.

* **The Method:** Use "Forward Chaining" (e.g., train on points $1$ through $t$, predict $t+1$).
* **The Metric:** Calculate the **Mean Absolute Scaled Error (MASE)** for both models.
* **The Refinement:** Assign a likelihood based on the ratio of the forecast errors. If the Logistic model consistently outperforms the Exponential model on the most recent "test" points, the S-curve is likely manifesting.

### 3. Log-Linear Residual Curvature
If we fit a straight line to $\ln(y)$, the residuals $\epsilon$ should be white noise for an exponential process. For a logistic process, the residuals will be **serially correlated** and exhibit **concavity**.

* **The Test:** Check the **Durbin-Watson statistic** or perform a **Runs Test** on the residuals.
* **The Refinement:** If the Durbin-Watson score deviates significantly from $2.0$ (indicating autocorrelation) and the residual plot shows a "frown" shape, the probability of an S-curve increases.

---

### Updated Python Implementation: Multi-Metric Ensemble

We can combine these into a "Scoring" system to reach a more robust posterior probability.

```python
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

def refined_posterior_analysis(t, y):
    # 1. BIC Evidence (as previously calculated)
    # Let's assume we have the BIC-derived probabilities:
    # p_bic_log = ... 
    
    # 2. Per-Capita Slope Evidence
    # Calculate per-capita growth: (dy/dt) / y
    dy = np.diff(y)
    dt = np.diff(t)
    per_capita_growth = dy / (dt * y[:-1])
    y_mid = y[:-1]
    
    # Regression: per_capita_growth ~ y_mid
    X = sm.add_constant(y_mid)
    model_pc = sm.OLS(per_capita_growth, X).fit()
    
    slope = model_pc.params[1]
    slope_std_err = model_pc.bse[1]
    
    # How likely is this slope if the true slope was 0 (Exponential)?
    # We calculate the Z-score for the slope being < 0
    z_score = slope / slope_std_err
    p_slope_evidence = norm.cdf(z_score) # Probability that slope is negative
    
    # 3. Residual Concavity Test
    # Fit log-linear: ln(y) = a + bt
    log_y = np.log(y)
    X_t = sm.add_constant(t)
    res_log_linear = sm.OLS(log_y, X_t).fit()
    residuals = res_log_linear.resid
    
    # Check for curvature by fitting a quadratic to the residuals: resid ~ t + t^2
    X_quad = np.column_stack([t, t**2])
    X_quad = sm.add_constant(X_quad)
    res_quad = sm.OLS(residuals, X_quad).fit()
    
    # If the coefficient for t^2 is significantly negative, it's 'concave down'
    quad_coeff = res_quad.params[2]
    p_concave_evidence = norm.cdf(quad_coeff / res_quad.bse[2])

    # 4. Bayesian Update
    # Start with 50/50 prior
    prior_log = 0.5
    
    # Multiply Likelihood Ratios (Simplified heuristic)
    # Using the p-values as proxies for evidence strength
    combined_score = (p_slope_evidence + p_concave_evidence) / 2
    
    return {
        "Slope_P_Value": model_pc.pvalues[1],
        "Quadratic_Residual_Coeff": quad_coeff,
        "S_Curve_Confidence": combined_score
    }
```

### 4. Qualitative Structural Indicators
Finally, as a data scientist, you should apply a **Bayesian Prior** based on domain constraints:

* **Total Addressable Market (TAM):** If $y$ is already $> 50\%$ of the known physical limit (e.g., global population, total internet users), the prior for the S-curve should be adjusted from $0.5$ to $0.9$.
* **The "Anti-Gravity" Factor:** In software (SaaS), marginal costs are near zero, allowing exponential phases to last much longer than in hardware or biological systems. In those cases, keep the exponential prior higher for longer.

---

### Summary of Refined Metrics

| Metric | Exponential Expectation | S-Curve (Logistic) Expectation |
| :--- | :--- | :--- |
| **Per-Capita Growth Slope** | $\beta_1 \approx 0$ | $\beta_1 < 0$ (Statistically Significant) |
| **Residual Curvature** | Random Noise | Concave Down ($t^2$ coefficient $< 0$) |
| **Durbin-Watson** | $\approx 2.0$ | $< 1.5$ (Positive Autocorrelation) |
| **Forecasting (CV)** | Lower MSE in early phase | Lower MSE as $y$ approaches $K/2$ |
