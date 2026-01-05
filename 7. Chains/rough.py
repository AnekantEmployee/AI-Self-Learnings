doc = '''{
  "notes": "Here are key points notes for the provided \"Comprehensive Guide to Linear Regression in Machine Learning\" document:\n\n---\n\n**Comprehensive Guide to Linear Regression in Machine Learning**\n*(Version 2025-07-21, Author: Kimi (Moonshot AI))*\n\n**Core Concept:**\n*   Models relationship between a continuous target (`y`) \nand predictors (`X`) using an affine function plus noise: `y = Xβ + ε`, where `ε` is typically `~ 𝒩 (0, σ²I)`.\n\n**1. What & Why**\n*   **Definition:** Linear relationship, continuous target, affine function + noise.\n*   **Use-Cases:** Baseline model, explainability/causal inference, calibration layer.\n*   **Limitations:** Assumes linearity in parameters, homoscedasticity, independence, normality (often relaxed); sensitive to outliers & multicollinearity; extrapolation risk.\n\n**2. Mathematical Foundations**\n*   **Design Matrix:** `X` (includes intercept).\n*   **Loss Function:** Mean Squared Error (MSE): `L(β) = (1/2n)‖y – Xβ‖²`.\n*   Gradient and Hessian are used for optimization.\n\n**3. Ordinary Least Squares (OLS)**\n*   **Closed-form Solution:** `β̂ = (XᵀX)⁻¹Xᵀy` (if `XᵀX` is invertible).\n*   **Computational Com\nplexity:** `O(np² + p³)`, stable via QR or SVD.\n*   **Statistical Properties (Gauss-Markov):** `β̂` is the Best Linear Unbiased Estimator (BLUE).\n\n**4. Probabilisti\nc Interpretations**\n*   **Maximum Likelihood Estimation (MLE):** Assuming Gaussian noise leads to OLS `β̂`.\n*   **Maximum A-Posteriori (MAP):** Placing Gaussian prio\nrs on `β` leads to Ridge regression.\n\n**5. Regularization (Ridge, Lasso, Elastic-Net)**\n*   **Purpose:** Addresses overfitting, multicollinearity, and enables feature selection.\n*   **Ridge (L2):** Adds `λ‖β‖₂²` to MSE loss; has closed-form solution.\n*   **Lasso (L1):** Adds `λ‖β‖₁` to MSE loss; induces sparsity (feature selection).\n*   **Elastic-Net:** Combines L1 and L2 penalties.\n*   **Hyperparameter `λ` Selection:** Typically via cross-validation or information criteria (AICc, BIC).\n\n**6. Feature Engineering & Transformations**\n*   **Standardization:** Numeric features (mean 0, var 1) for regularized models.\n*   **Categorical Encoding:** One-hot encoding (drop one level to avoid dummy trap).\n*   **Target Transformation:** Log/Box-Cox for skewed targets.\n*   **Non-linear Relationships:** B-splines or piecewise linear splines.\n*   **Interaction Terms:** Product of features (`xᵢ · xⱼ`).\n\n**7. Diagnostics & Assumptions**\n*   **Residual Plots:**\n    *   Vs. fitted values: Check homoscedasticity, linearity.\n    *   Q-Q plot: Check normality of residuals.\n*   **Influence Measures:** Leverage, Cook’s distance, DFBETAS (identify influential points/outliers).\n*   **Statistical Tests:** Breusch-Pagan (heteroscedasticity), Durbin-Watson (serial correlation), VIF (multicollinearity).\n\n**8. Multicollinearity & Remedies**\n*   **Symptoms:** Large standard errors, unstable coefficients, high VIF (>10).\n*   **Remedies:** Drop redundant features, Principal Component Regression (PCR), Partial Least Squares (PLS), Regularization (especially Ridge).\n\n**9. Polynomial & Interaction Models**\n*   **Polynomials:** Add `x, x², ..., xᵈ` terms; can lead to ill-conditioning.\n*   **Interactions:** Model synergistic effects (e.g., `x₁x₂`).\n\n**10. Robust Regression**\n*   **Purpose:** Less sensitive to outliers.\n*   **Methods:** Huber loss, RANSAC, Theil-Sen, Least Trimmed Squares (LTS).\n\n**11. Bayesian Linear Regression**\n*   **Approach:** Incorporates prior beliefs about parameters to derive posterior distributions.\n*   **Benefits:** Provides full predictive distribution and natural uncertainty quantification.\n*   Uses MCMC \nfor complex models.\n\n**12. Online / Incremental Learning**\n*   **Purpose:** Update model with new data without retraining on the entire dataset.\n*   **Recursive Least Squares (RLS):** `O(p²)` per step.\n*   **Alternative:** Stochastic Gradient Descent (SGD).\n\n**13. Evaluation & Model Selection**\n*   **Metrics:** RMSE, MAE, R², Adjusted R², MAPE/sMAPE.\n*   **Cross-validation:** k-fold, Repeated CV, TimeSeriesSplit (for temporal data).\n*   **Model Comparison:** Nested CV, Likelihood-ratio \ntest, AICc/BIC.\n\n**14. Implementation Cheat-Sheet (Python & R)**\n*   **Python:** `scikit-learn` (LinearRegression, RidgeCV, LassoCV, Pipeline, StandardScaler, PolynomialFeatures, GridSearchCV), `statsmodels` (for statistical inference/summary).\n*   **R:** `tidymodels` (linear_reg, workflow), `glmnet` (for regularized models).\n\n**15. Production Deployment Checklist**\n*   Retrain on full data with best hyperparameters.\n*   Save data transformers (scalers, encoders) with the model.\n*   Monitor model drift (residual mean/variance).\n*   Provide prediction intervals.\n*   Document feature definitions, training window, assumptions.\n*   Implement unit tests.\n\n**16. Frequently Asked Questions (FAQs)**\n*   **Standardization:** Mandatory for regularized models; optional for OLS.\n*   **Observations per Parameter:** ≥ 10-20 samples for stable OLS; regularization relaxes this.\n*   **High R² but poor test RMSE:** Indicates overfitting or data leakage.\n*   **LR for Classification:** Technically possible, but Logistic Regression is preferred for probabilistic outputs.\n*   **Missing Data:** Use Multiple Imputation (MICE) or model-based imputation.\n\n**17. References & Further Reading**\n*   Key textbooks: \"The Elements of Statistical Learning\", \"Bayesian Data Analysis\", \"Applied Linear Statistical Models\".\n*   Online resources: scikit-learn docs, \"Forecasting: Principles and Practice\".\n\n---",
  "quiz": {
    "quiz_title": "Linear Regression in Machine Learning",
    "questions": [
      {
        "question_number": 1,
        "question_text": "According to the document, what is the fundamental assumption linear regression makes about the relationship between the continuous target 'y' and predictors 'X'?",
        "options": {
          "A": "A logarithmic function plus noise.",
          "B": "An exponential function plus noise.",
          "C": "An affine function plus noise.",
          "D": "A polynomial function with infinite degree."
        },
        "correct_answer": "C",
        "reference": "Section 1: WHAT & WHY - Definition"
      },
      {
        "question_number": 2,
        "question_text": "What is the closed-form solution for the Ordinary Least Squares (OLS) estimator β̂, assuming XᵀX is invertible, as provided in the document?",
        "options": {
          "A": "β̂ = (XᵀX)Xᵀy",
          "B": "β̂ = (XᵀX)⁻¹Xᵀy",
          "C": "β̂ = (Xᵀy)⁻¹XᵀX",
          "D": "β̂ = (XᵀX)⁻¹yXᵀ"
        },
        "correct_answer": "B",
        "reference": "Section 3: ORDINARY LEAST SQUARES (OLS)"
      },
      {
        "question_number": 3,
        "question_text": "Which type of regularization is specifically mentioned in the document as inducing sparsity in the model coefficients?",
        "options": {
          "A": "Ridge (L2) regression.",
          "B": "Elastic-Net regression.",
          "C": "Lasso (L1) regression.",
          "D": "Bayesian Linear Regression."
        },
        "correct_answer": "C",
        "reference": "Section 5: REGULARIZATION - Lasso (L1)"
      },
      {
        "question_number": 4,
        "question_text": "According to the document, which of the following is a common symptom of \nmulticollinearity?",
        "options": {
          "A": "Small standard errors and stable coefficient signs.",
          "B": "High R-squared but low VIF values.",
          "C": "Large standard errors, unstable signs, and high VIF (>10).",
          "D": "Normally distributed residuals."
        },
        "correct_answer": "C",
        "reference": "Section 8: MULTICOLLINEARITY & REMEDIES - Symptoms"
      },
      {
        "question_number": 5,
        "question_text": "Based on the 'Frequently Asked Questions' section, when is standardizing numeric features considered mandatory?",
        "options": {
          "A": "Always, regardless of the model type.",
          "B": "Only for OLS models to improve interpretability.",
          "C": "For Ridge, Lasso, and Elastic-Net regularization.",
          "D": "Never, as it can remove important information."
        },
        "correct_answer": "C",
        "reference": "Section 16: FREQUENTLY ASKED QUESTIONS - Q1"
      }
    ]
  }
}'''
print(doc)