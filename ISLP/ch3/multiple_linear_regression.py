import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import ModelSpec, summarize, poly
import matplotlib.pyplot as plt
import seaborn as sns

def print_stats(model, name):
    print("="*50)
    print(f"stats for model: {name}")
    print("=")
    print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
    print(f"F-statistic: {model.fvalue:.2f}, p-value: {model.f_pvalue:.4e}")
    print(f"Residual standard error: {np.sqrt(model.mse_resid):.3f}")
    print(summarize(model))
    print("="*50)

Boston = load_data('Boston')
Boston = Boston.sort_values(by='medv').reset_index(drop=True)

print(Boston)
y = Boston['medv']

terms = Boston.columns.drop('medv')
design = ModelSpec(terms)
X = design.fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
print_stats(results1, "all variables")

print("vifs:")
vals = [VIF(X, i) for i in range(1, X.shape [1])]
vif = pd.DataFrame ({'vif':vals}, index=X.columns [1:])
print(vif)

# medv removed bc we cant have predicted value as prediction, it would defeat the purpouse
# age and indus removed, because p-value too big
terms = Boston.columns.drop(['medv', 'age', 'indus']) 
design = ModelSpec(terms)
X = design.fit_transform(Boston)
modelA = sm.OLS(y, X)
resultsA = modelA.fit()
print_stats(resultsA, "age and indus dropped")

all_columns = Boston.columns.drop(['medv', 'age', 'indus']).tolist()
variables_for_poly = ['crim', 'zn', 'rm', 'dis', 'ptratio', 'lstat']

linear_terms = [col for col in all_columns if col not in variables_for_poly]

poly_terms = [poly(var, degree=2) for var in variables_for_poly]
terms = linear_terms + poly_terms
design = ModelSpec(terms)
X = design.fit_transform(Boston)
modelA = sm.OLS(y, X)
resultsA = modelA.fit()
print_stats(resultsA, "age and indus dropped, some squared terms")

'''
# tax removed, bc it has very big correlation (like 0.91) with rad, and it also has viff at like 7.25 (which is bigger than 5)
terms = Boston.columns.drop(['medv', 'age', 'indus', 'tax']) 
design = ModelSpec(terms)
X = design.fit_transform(Boston)
modelB = sm.OLS(y, X)
resultsB = modelB.fit()
print_stats(resultsB, "age and indus AND AlSO tax dropped") 
# turns out this model is slightly worse than resultsA, bc r squared dropped and RSE increased
'''


plt.figure()
plt.scatter(Boston.index, Boston['medv'], alpha=0.5)
plt.scatter(Boston.index, results1.fittedvalues, alpha=0.5)
plt.xlabel('index')
plt.title('medv guesses for each index, vs real values, for sorted dataset')
plt.ylabel('medv')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(results1.fittedvalues, results1.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# Scatter plots for each predictor
# crim, zn, rm, nox, dis, rad, tax, ptratio and lstat look to me like they are cureved, might need x^2 param?
# Scatter plots for all variables against medv
predictors = Boston.columns.drop('medv')
fig, axes = plt.subplots(5, 5, figsize=(16, 16))
axes = axes.ravel()

for i, p in enumerate(predictors):
    axes[i].scatter(Boston[p], Boston['medv'], alpha=0.2)
    axes[i].set_xlabel(p)
    axes[i].set_ylabel('medv')

plt.tight_layout()
plt.show()

# correlation matrix heatmap nice image
plt.figure(figsize=(10, 8))
sns.heatmap(Boston.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()