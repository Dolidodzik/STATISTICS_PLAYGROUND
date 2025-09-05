import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP.models import ModelSpec, summarize, poly


Smarket = load_data('Smarket')
Smarket = pd.get_dummies(Smarket, columns=['Direction'])
print(Smarket)
print(Smarket.columns)
print(Smarket.dtypes)
print(Smarket.corr())
plt.figure(figsize=(10, 8))
sns.heatmap(Smarket.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

Smarket.plot(y='Volume')
plt.show()

terms = Smarket.columns.drop(['Today', 'Year', 'Direction_Down', 'Direction_Up'])
design = ModelSpec(terms)
X = design.fit_transform(Smarket)
y = Smarket.Direction_Up
glm = sm.GLM(y, X, family=sm.families.Binomial())
results = glm.fit()
print(summarize(results))