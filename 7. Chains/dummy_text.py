document = """
────────────────────────────────────────────────────────
Comprehensive Guide to Linear Regression in Machine Learning
────────────────────────────────────────────────────────
Version 2025-07-21  •  Author: Kimi (Moonshot AI)

CONTENTS
1.  What & Why  
2.  Mathematical Foundations  
3.  Ordinary Least Squares (OLS)  
4.  Probabilistic Interpretations  
5.  Regularization (Ridge, Lasso, Elastic-Net)  
6.  Feature Engineering & Transformations  
7.  Diagnostics & Assumptions  
8.  Multicollinearity & Remedies  
9.  Polynomial & Interaction Models  
10. Robust Regression  
11.Bayesian Linear Regression  
12.Online / Incremental Learning  
13.Evaluation & Model Selection  
14.Implementation Cheat-Sheet (Python & R)  
15.Checklist for Production Deployment  
16.Frequently Asked Questions  
17.References & Further Reading  

────────────────────────────────────────
1. WHAT & WHY
────────────────────────────────────────
Definition  
Linear regression models the relationship between a continuous target y ∈ ℝ and one or more predictors X = [x₁, …, x_p] by assuming an affine function plus noise:

  y = Xβ + ε, ε ~ 𝒩(0, σ²I)

Use-Cases  
• Baseline for tabular regression problems  
• Explainability & causal inference (instrumental variables, difference-in-differences)  
• Calibration layer in complex pipelines (e.g., residual correction after tree models)  

Limitations  
• Linearity in parameters, homoscedasticity, independence, normality (often relaxed)  
• Sensitive to outliers & multicollinearity  
• Extrapolation risk outside training support  

────────────────────────────────────────
2. MATHEMATICAL FOUNDATIONS
────────────────────────────────────────
Design matrix X ∈ ℝ^{n×(p+1)} with intercept column of 1s.  
Loss = Mean Squared Error (MSE):

  L(β) = (1/2n)‖y – Xβ‖²

Gradient & Hessian  
 ∇L = –(1/n)Xᵀ(y – Xβ)  
 H = (1/n)XᵀX ⪰ 0 (positive semi-definite)

────────────────────────────────────────
3. ORDINARY LEAST SQUARES (OLS)
────────────────────────────────────────
Closed-form solution  
 β̂ = (XᵀX)^{-1}Xᵀy  (only if XᵀX invertible)

Computational complexity: O(np² + p³). Numerically stable via QR or SVD.

Statistical properties under Gauss-Markov assumptions  
• β̂ is BLUE (Best Linear Unbiased Estimator)  
• Var(β̂) = σ²(XᵀX)^{-1}  
• σ̂² = RSS / (n – p – 1)  

────────────────────────────────────────
4. PROBABILISTIC INTERPRETATIONS
────────────────────────────────────────
Maximum Likelihood Estimation (MLE)  
Assume y|X,β,σ² ~ 𝒩(Xβ, σ²I). Maximizing log-likelihood yields the same β̂ as OLS.

Maximum A-Posteriori (MAP)  
Place priors: β ~ 𝒩(0, τ²I). MAP estimate becomes Ridge regression.

────────────────────────────────────────
5. REGULARIZATION
────────────────────────────────────────
Ridge (L2)  
 β̂_ridge = argmin ‖y – Xβ‖² + λ‖β‖₂²  
Closed-form: (XᵀX + λI)^{-1}Xᵀy

Lasso (L1)  
 β̂_lasso = argmin ‖y – Xβ‖² + λ‖β‖₁  
Induces sparsity; solved via coordinate descent or LARS.

Elastic-Net  
 λ₁‖β‖₁ + λ₂‖β‖₂² balances sparsity & grouping effects.

Choosing λ via cross-validation or information criteria (AICc, BIC, GCV).

────────────────────────────────────────
6. FEATURE ENGINEERING & TRANSFORMATIONS
────────────────────────────────────────
• Standardize numeric features (mean 0, var 1) for regularized models.  
• One-hot encode categoricals; drop one level to avoid dummy trap.  
• Log / Box-Cox transform skewed targets.  
• B-splines or piecewise linear splines for non-linear relationships while staying linear in parameters.  
• Interaction terms: x_i · x_j. Avoid combinatorial explosion with Lasso.

────────────────────────────────────────
7. DIAGNOSTICS & ASSUMPTIONS
────────────────────────────────────────
Residual plots  
• vs. fitted: check homoscedasticity & non-linearity.  
• Q-Q plot: normality of residuals (important for inference).  
Scale-Location plot: detect heteroscedasticity.

Influence measures  
• Leverage h_ii = x_iᵀ(XᵀX)^{-1}x_i. Points with h_ii > 2(p+1)/n are high leverage.  
• Cook’s distance: combine leverage & residual magnitude.  
• DFBETAS: per-coefficient influence.

Breusch-Pagan test (heteroscedasticity), Durbin-Watson (serial correlation), VIF (multicollinearity).

────────────────────────────────────────
8. MULTICOLLINEARITY & REMEDIES
────────────────────────────────────────
Symptoms  
• Large standard errors, unstable signs, high VIF (>10).  
Remedies  
• Drop redundant features.  
• Principal Component Regression (PCR) or Partial Least Squares (PLS).  
• Regularization (Ridge especially).

────────────────────────────────────────
9. POLYNOMIAL & INTERACTION MODELS
────────────────────────────────────────
Polynomial degree d adds columns x_i, x_i², …, x_i^d.  
Matrix becomes ill-conditioned quickly → use orthogonal polynomials or regularization.

Interaction example: y = β₀ + β₁x₁ + β₂x₂ + β₃x₁x₂ + ε.

────────────────────────────────────────
10. ROBUST REGRESSION
────────────────────────────────────────
• Huber loss: quadratic near 0, linear in tails.  
• RANSAC: fit on consensus subset.  
• Theil-Sen or Least Trimmed Squares (LTS) for high breakdown points.

────────────────────────────────────────
11. BAYESIAN LINEAR REGRESSION
────────────────────────────────────────
Conjugate priors  
β|σ² ~ 𝒩(μ₀, σ²V₀)  
σ² ~ IG(a₀, b₀)

Posterior  
β|y,σ² ~ 𝒩(μ_n, σ²V_n)  
σ²|y ~ IG(a_n, b_n)

Closed-form updates available; use MCMC (NUTS) for non-conjugate or hierarchical extensions.

Benefits: full predictive distribution, natural uncertainty quantification.

────────────────────────────────────────
12. ONLINE / INCREMENTAL LEARNING
────────────────────────────────────────
Recursive Least Squares (RLS)  
Initialize P₀ = δI, β₀ = 0  
For each new sample (x_t, y_t):  
 e_t = y_t – x_tᵀβ_{t-1}  
 γ_t = (1 + x_tᵀP_{t-1}x_t)^{-1}  
 β_t = β_{t-1} + γ_t P_{t-1}x_t e_t  
 P_t = P_{t-1} – γ_t P_{t-1}x_t x_tᵀP_{t-1}

Complexity O(p²) per step, no need to store full data.

Alternative: Stochastic Gradient Descent (SGD) with learning rate schedules.

────────────────────────────────────────
13. EVALUATION & MODEL SELECTION
────────────────────────────────────────
Metrics  
• RMSE (scale-dependent)  
• MAE (robust to outliers)  
• R² & Adjusted R² (for nested models)  
• MAPE / sMAPE (when relative errors matter)

Cross-validation  
• k-fold (k=5 or 10)  
• Repeated CV for small data  
• TimeSeriesSplit for temporal data

Model comparison  
• Nested CV to avoid optimistic bias.  
• Likelihood-ratio test for nested OLS models.  
• AICc or BIC for penalized likelihood balance.

────────────────────────────────────────
14. IMPLEMENTATION CHEAT-SHEET
────────────────────────────────────────
Python (scikit-learn)

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Basic OLS
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Regularized pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', RidgeCV(alphas=np.logspace(-3,3,100)))
])
pipe.fit(X_train, y_train)

# Statsmodels for inference
import statsmodels.api as sm
X_const = sm.add_constant(X)
ols = sm.OLS(y, X_const).fit()
print(ols.summary())

R (tidymodels)

library(tidymodels)
spec_lin <- linear_reg() %>% set_engine("lm")
wf <- workflow() %>% add_model(spec_lin) %>% add_formula(y ~ .)
fit <- fit(wf, data = train)
predict(fit, test)

Elastic-net via glmnet:

library(glmnet)
cvfit <- cv.glmnet(as.matrix(X), y, alpha = 0.5)  # 0=ridge, 1=lasso
coef(cvfit, s = "lambda.min")

────────────────────────────────────────
15. PRODUCTION DEPLOYMENT CHECKLIST
────────────────────────────────────────
☐ Re-train on full data with best hyper-parameters.  
☐ Save scaler / polynomial transformer with model (sklearn.pipeline or vetiver in R).  
☐ Monitor drift: track residual mean & variance on new data; trigger retrain if >3σ shift.  
☐ Provide prediction intervals: analytic for OLS, bootstrap for robust & non-linear extensions.  
☐ Document feature definitions, training window, and assumptions.  
☐ Unit tests: intercept-only model should yield mean(y); gradient checks for custom losses.

────────────────────────────────────────
16. FREQUENTLY ASKED QUESTIONS
────────────────────────────────────────
Q1. Should I always standardize features?  
• Mandatory for Ridge/Lasso/Elastic-Net. Optional but harmless for OLS.

Q2. How many observations per parameter?  
• Rule of thumb ≥ 10–20 samples per coefficient for stable OLS; regularization relaxes this.

Q3. High R² but poor test RMSE?  
• Overfitting due to many features or data leakage. Use nested CV and check residual plots.

Q4. Can I use linear regression for classification?  
• Technically yes (LDA is equivalent to linear regression on class indicators), but logistic regression is preferred for probabilistic outputs.

Q5. Handling missing data?  
• Multiple Imputation by Chained Equations (MICE) or model-based imputation; avoid list-wise deletion unless MCAR.

────────────────────────────────────────
17. REFERENCES & FURTHER READING
────────────────────────────────────────
• Hastie, Tibshirani, Friedman – “The Elements of Statistical Learning”, Ch 3  
• Gelman et al. – “Bayesian Data Analysis”, Ch 14–16  
• Kutner, Nachtsheim, Neter – “Applied Linear Statistical Models”  
• scikit-learn documentation: Linear Models section  
• Gelman & Hill – “Data Analysis Using Regression and Multilevel/Hierarchical Models”  
• Hyndman & Athanasopoulos – “Forecasting: Principles and Practice” (online, free)  

End of document"""