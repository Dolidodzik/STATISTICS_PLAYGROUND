import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('validation.csv')
print("Loaded training and test dfs")
print("train_df shape: ", train_df.shape)
print("test_df shape: ", test_df.shape)

X_train = train_df.drop('price', axis=1)
y_train = train_df['price']
X_test = test_df.drop('price', axis=1)
y_test = test_df['price']

categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', KNNImputer(n_neighbors=3)) # KNN after scaler, so KNN works on scaled data as it needs
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

print("making sure preprocessing works: ")
X_train_processed = preprocessor.fit_transform(X_train)
X_train_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
print(X_train_df.describe()) 

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-5, 5, 100)

print("DOING LASSO: ")
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso())
])

cv_scores = []
for alpha in alphas:
    lasso_pipeline.set_params(regressor__alpha=alpha)
    scores = cross_val_score(lasso_pipeline, X_train, y_train, 
                           cv=kfold, 
                           scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

optimal_alpha = alphas[np.argmin(cv_scores)]
print(f'Optimal alpha: {optimal_alpha}')

lasso_pipeline.set_params(regressor__alpha=optimal_alpha)
lasso_pipeline.fit(X_train, y_train)

# running the lasso on train set
y_pred_test = lasso_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Final MSE: {mse:.3f}")
print(f"Final R²: {r2:.3f}")

plt.semilogx(alphas, cv_scores)
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('LASSO Cross-Validation Results')
plt.axvline(optimal_alpha, color='red', linestyle='--')
plt.show()

print("lasso coefficeints: ")
lasso_model = lasso_pipeline.named_steps['regressor']
coefficients = lasso_model.coef_

feature_names = lasso_pipeline[:-1].get_feature_names_out()
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
})

print(f"All coefficients count: {coef_df.shape[0]}")
nonzero_coef_df = coef_df[coef_df['coefficient'] != 0]
print(f"non zero coefficeints (there are {nonzero_coef_df.shape[0]} of them): ")
print(nonzero_coef_df)

print("===============================")
print("doing ridge")