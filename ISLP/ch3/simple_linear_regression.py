import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import ModelSpec, summarize, poly
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Boston housing dataset
Boston = load_data("Boston")
print(Boston)

# Add squared term for lstat
Boston['lstat_sq'] = Boston['lstat'] ** 2

# Set up quadratic regression model with 'lstat' and 'lstat_sq' as predictors
design = ModelSpec(['lstat', 'lstat_sq'])
X = design.fit_transform(Boston)
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))

# Make predictions for new values of lstat
new_df = pd.DataFrame({'lstat':[5, 10, 15, 99]})
new_df['lstat_sq'] = new_df['lstat'] ** 2
newX = design.transform(new_df)
print(newX)

# Get predictions and confidence intervals for these new values
new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print(new_predictions.conf_int(alpha=0.99))

# Create a scatter plot of lstat vs medv and add the quadratic regression curve
ax = Boston.plot.scatter('lstat', 'medv')

# Generate points for the quadratic curve
x_range = np.linspace(Boston['lstat'].min(), Boston['lstat'].max(), 100)
x_range_df = pd.DataFrame({'lstat': x_range, 'lstat_sq': x_range**2})
x_range_design = design.transform(x_range_df)
y_pred = results.predict(x_range_design)

ax.plot(x_range, y_pred, color='red')
plt.show()

# DIAGNOSTIC PLOTS

# Create a residual plot
fig, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

# Calculate leverage statistics
infl = results.get_influence()
fig, ax = subplots(figsize=(8,8))
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
max_leverage_index = np.argmax(infl.hat_matrix_diag)
plt.show()
print("max_leverage_index: ", max_leverage_index)

# MODEL SUMMARY
print("\n" + "="*60)
print("QUADRATIC MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"R-squared: {results.rsquared:.4f}")
print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
print(f"F-statistic: {results.fvalue:.2f} (p-value: {results.f_pvalue:.4e})")
print(f"Akaike Information Criterion (AIC): {results.aic:.2f}")
print(f"Bayesian Information Criterion (BIC): {results.bic:.2f}")
print("="*60)
print("COEFFICIENT SUMMARY")
print("="*60)
for i, param in enumerate(results.params.index):
    print(f"{param:12}: {results.params[i]:8.4f} (p-value: {results.pvalues[i]:8.4e})")
print("="*60)

# Compare with linear model
design_linear = ModelSpec(['lstat'])
X_linear = design_linear.fit_transform(Boston)
model_linear = sm.OLS(y, X_linear)
results_linear = model_linear.fit()

print("\n" + "="*60)
print("COMPARISON: LINEAR vs QUADRATIC MODEL")
print("="*60)
print(f"{'Metric':20} {'Linear':>10} {'Quadratic':>10}")
print(f"{'R-squared':20} {results_linear.rsquared:10.4f} {results.rsquared:10.4f}")
print(f"{'Adjusted R-squared':20} {results_linear.rsquared_adj:10.4f} {results.rsquared_adj:10.4f}")
print(f"{'AIC':20} {results_linear.aic:10.2f} {results.aic:10.2f}")
print(f"{'BIC':20} {results_linear.bic:10.2f} {results.bic:10.2f}")
print("="*60)



# for model with just 2 params, intercept and linear coefficient
'''
============================================================
MODEL PERFORMANCE SUMMARY
============================================================
R-squared: 0.5441
Adjusted R-squared: 0.5432
F-statistic: 601.62 (p-value: 5.0811e-88)
Akaike Information Criterion (AIC): 3286.97
Bayesian Information Criterion (BIC): 3295.43
============================================================
COEFFICIENT SUMMARY
============================================================
/home/lesnik/coding/ISLP/ch3/simple_linear_regression.py:77: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  print(f"{param:12}: {results.params[i]:8.4f} (p-value: {results.pvalues[i]:8.4e})")
intercept   :  34.5538 (p-value: 3.7431e-236)
lstat       :  -0.9500 (p-value: 5.0811e-88)
============================================================
'''