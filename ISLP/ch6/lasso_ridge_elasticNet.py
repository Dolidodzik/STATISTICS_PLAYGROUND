import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score



students_df =  pd.read_csv('student-mat.csv')

'''
# this shows us that G1, G2, G3 are very highly correlated, that is because they all are grades from different periods
# hence we drop G1 and G2, and only focus on G3, the final grade
plt.figure(figsize=(10, 8))
heatmap_df = students_df.drop(students_df.select_dtypes(exclude='number').columns.tolist(), axis=1)
sns.heatmap(heatmap_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
'''
students_df.drop(['G1', 'G2'], axis=1)
print("students df after dropping G1 and G2")
print(students_df)

X = students_df.drop('G3', axis=1)
y = students_df['G3']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso())
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

alphas = np.logspace(-4, 4, 100)
print("alphas: ", alphas)

cv_scores = []
for alpha in alphas:
    pipeline.set_params(regressor__alpha=alpha)
    scores = cross_val_score(pipeline, X, y, 
                           cv=kfold, 
                           scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

optimal_alpha = alphas[np.argmin(cv_scores)]
print(f'Optimal alpha: {optimal_alpha}')

pipeline.set_params(regressor__alpha=optimal_alpha)
pipeline.fit(X, y)


y_pred = pipeline.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Final MSE: {mse:.2f}")
print(f"Final R²: {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, cv_scores)
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('LASSO Cross-Validation Results')
plt.axvline(optimal_alpha, color='red', linestyle='--')
plt.show()