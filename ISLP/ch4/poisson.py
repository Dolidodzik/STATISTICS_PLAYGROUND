import pandas as pd
import numpy as np
from ISLP import load_data
from ISLP.models import ModelSpec, summarize, contrast
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots


Bikeshare = load_data('Bikeshare')
print(Bikeshare)

X = ModelSpec(['mnth', 'hr', 'workingday', 'temp', 'weathersit']).fit_transform(Bikeshare)
Y = Bikeshare['bikers']
M_lm = sm.OLS(Y, X).fit()
print(summarize(M_lm))


hr_encode = contrast('hr', 'sum')
mnth_encode = contrast('mnth', 'sum')
X2 = ModelSpec([
    mnth_encode,
    hr_encode,
    'workingday',
    'temp',
    'weathersit'
]).fit_transform(Bikeshare)
M2_lm = sm.OLS(Y, X2).fit()
S2 = summarize(M2_lm)
print(S2)
print(type(S2))

coef_month = S2[S2.index.str.contains('mnth')]['coef']

months = Bikeshare['mnth'].dtype.categories
coef_month = pd.concat([
    coef_month,
    pd.Series([-coef_month.sum()], index=['mnth[Dec]'])
])

print("================================================")
print("coef month with December")
print(coef_month)

fig_month, ax_month = plt.subplots(figsize=(8, 8))
x_month = np.arange(coef_month.shape[0])
ax_month.plot(x_month, coef_month, marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)
plt.title('linear regression months')
plt.show()

coef_hr = S2[S2.index.str.contains('hr')]['coef']
coef_hr = coef_hr.reindex(['hr[{0}]'.format(h) for h in range(23)])
coef_hr = pd.concat([
    coef_hr,
    pd.Series([-coef_hr.sum()], index=['hr[23]'])
])

fig_hr, ax_hr = plt.subplots(figsize=(8, 8))
x_hr = np.arange(coef_hr.shape[0])
ax_hr.plot(x_hr, coef_hr, marker='o', ms=10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20)
plt.title('linear regression hours')
plt.show()


############### POISSON
M_pois = sm.GLM(Y, X2, family=sm.families.Poisson()).fit()
S_pois = summarize(M_pois)

coef_month = S_pois[S_pois.index.str.contains('mnth')]['coef']
coef_month = pd.concat([
    coef_month,
    pd.Series([-coef_month.sum()], index=['mnth[Dec]'])
])

coef_hr = S_pois[S_pois.index.str.contains('hr')]['coef']
coef_hr = coef_hr.reindex(['hr[{0}]'.format(h) for h in range(23)])
coef_hr = pd.concat([
    coef_hr,
    pd.Series([-coef_hr.sum()], index=['hr[23]'])
])

fig_pois, (ax_month, ax_hr) = plt.subplots(1, 2, figsize=(16, 8))

x_month = np.arange(coef_month.shape[0])
ax_month.plot(x_month, coef_month, marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)

x_hr = np.arange(coef_hr.shape[0])
ax_hr.plot(x_hr, coef_hr, marker='o', ms=10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20)

plt.tight_layout()
plt.title('poisson hour and month')
plt.show()

# Compare fitted values between linear regression and Poisson regression
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(M2_lm.fittedvalues, M_pois.fittedvalues, s=20)
ax.set_xlabel('Linear Regression Fit', fontsize=20)
ax.set_ylabel('Poisson Regression Fit', fontsize=20)
ax.axline([0, 0], c='black', linewidth=3, linestyle='--', slope=1)
plt.title('linear vs poisson')
plt.show()

