import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin


# useful for targeted transofrmations etc that we know from EDA that might be useful
class SmartFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['dist2', 'dist3', 'dist4']
        self.reciprocal_cols = ['crime_rate', 'poor_prop']
        self.log_cols = ['dist1', 'parks']
        self.square_cols = []

        self.sqrt_cols = []

        '''
        self.columns_to_drop = []
        self.reciprocal_cols = []
        self.log_cols = []
        self.square_cols = []
        '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        X = X.drop([col for col in self.columns_to_drop if col in X.columns], axis=1)
        
        for col in self.reciprocal_cols:
            if col in X.columns:
                X[col] = np.where(X[col] > 1e-10, 1/X[col], 0)
        
        for col in self.log_cols:
            if col in X.columns:
                X[col] = np.log(np.clip(X[col], 1e-10, None))
        
        for col in self.square_cols:
            if col in X.columns:
                X[col] = X[col]**2
                
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f for f in input_features if f not in self.columns_to_drop]

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

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('feature_engineer', SmartFeatureEngineer()),
        ('imputer', KNNImputer(n_neighbors=3)),
        ('scaler', StandardScaler())
    ]), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
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
print("----")
print(f"Final MSE: {mse:.3f}")
print(f"Final R²: {r2:.3f}")
print("----")

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

'''
lasso performance without transformations
Final MSE: 24.707
Final R²: 0.649
=
lasso performance with too many transformations (x, x^2, log(x), 1/x, sqrt(x) for every numerical pred)
Final MSE: 43.202
Final R²: 0.387
so adding too many transformations like a monkey doesn't work with lasso!
=
lasso with x, x^2, sqrt(x) transformations
Final MSE: 46.612
Final R²: 0.338
horrible, we should be careful with adding any transformations as they can lead to noise, overfitting, curse of dimensionality and other bad stuff
=
lasso with specific transformations only for specific columns, selected manually by looking at scatterplots of every predictor vs price
Final MSE: 18.834
Final R²: 0.733
GREAT! Much better than without transformations! it turns out adding age to 1/x is slighly worse than not adding it, 0.732 vs 0.733
'''

print("===============================")
print("doing ridge")